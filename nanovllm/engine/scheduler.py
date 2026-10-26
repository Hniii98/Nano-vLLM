from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
            这里固定了 nano-vllm 调度策略：优先 prefill, 其次 decode。

            初次添加的 seq 会被放到 waitting 序列中，因此 waiting 序列中都是等待 prefill的 seqs。
            调度器每次先去取 waitting 中的 seq，一旦取到就直接返回，不会交叉 waiting 和 running。

            对于 prefill 部分，会先计算缓存命中的块数，然后根据本轮调度的额度计算出可用的调度长度记录
            到 seq 中，之后 seq 就可以根据缓存的长度和调度的长度去做切片进行计算。
        """

        scheduled_seqs = []
        num_accum_scheduled_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            token_budget = self.max_num_batched_tokens - num_accum_scheduled_tokens

            if token_budget == 0:
                break

            first_time_prefill = not seq.block_table

            if first_time_prefill:
                num_cache_hit_blocks = self.block_manager.can_allocate(seq)
                if num_cache_hit_blocks == -1:
                    break
                num_cache_hit_tokens = num_cache_hit_blocks * self.block_size
                num_pending_tokens = seq.num_tokens - num_cache_hit_tokens
            else:
                num_pending_tokens = seq.num_tokens - seq.num_cached_tokens

            first_scheduled_seq = not scheduled_seqs
            if token_budget < num_pending_tokens and not first_scheduled_seq:  # only allow chunked prefill for the first seq
                break
            if first_time_prefill:
                self.block_manager.allocate(seq, num_cache_hit_blocks)

            seq.num_scheduled_tokens = min(num_pending_tokens, token_budget)
            num_accum_scheduled_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
