"""Dictionary trie and dynamic-programming route selection for jieba."""

from std.algorithm import parallelize
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 32_768
comptime MAX_WORKERS = 8


def find_child(
    node: Int,
    codepoint: UInt32,
    edge_offsets: IPtr,
    edge_chars: U32Ptr,
    edge_children: IPtr,
    root_children: IPtr,
) -> Int:
    if node == 0:
        if codepoint <= UInt32(65535):
            return Int(root_children[Int(codepoint)])
    var lo = Int(edge_offsets[node])
    var hi = Int(edge_offsets[node + 1])
    if hi - lo <= 8:
        comptime W = simdwidthof[DType.float64]()
        if hi - lo < W:
            for index in range(lo, hi):
                if edge_chars[index] == codepoint:
                    return Int(edge_children[index])
            return -1
        var target = SIMD[DType.uint32, W](codepoint)
        var ones = SIMD[DType.uint32, W](1)
        var zeros = SIMD[DType.uint32, W](0)
        var index = lo
        while index + W <= hi:
            var values = edge_chars.load[width=W](index)
            var matches = values.eq(target).select(ones, zeros)
            if Int(matches.reduce_add()[0]) != 0:
                for lane in range(W):
                    if values[lane] == codepoint:
                        return Int(edge_children[index + lane])
            index += W
        while index < hi:
            if edge_chars[index] == codepoint:
                return Int(edge_children[index])
            index += 1
        return -1
    while lo < hi:
        var middle = (lo + hi) // 2
        if edge_chars[middle] < codepoint:
            lo = middle + 1
        else:
            hi = middle
    if (
        lo < Int(edge_offsets[node + 1])
        and edge_chars[lo] == codepoint
    ):
        return Int(edge_children[lo])
    return -1


def route_range(
    chars: U32Ptr,
    start: Int,
    end: Int,
    edge_chars: U32Ptr,
    edge_offsets: IPtr,
    edge_children: IPtr,
    node_weights: FPtr,
    root_children: IPtr,
    max_word_length: Int,
    unknown_weight: Float64,
    route: IPtr,
    scores: FPtr,
):
    scores[end] = 0.0
    for reverse_idx in range(end - start):
        var idx = end - reverse_idx - 1
        var best_end = idx
        var best_score = unknown_weight + scores[idx + 1]
        var limit = min(end, idx + max_word_length)
        var node = 0
        for candidate_end in range(idx + 1, limit + 1):
            node = find_child(
                node,
                chars[candidate_end - 1],
                edge_offsets,
                edge_chars,
                edge_children,
                root_children,
            )
            if node < 0:
                break
            if node_weights[node] > -1.0e290:
                var candidate_score = node_weights[node] + scores[candidate_end]
                if (
                    candidate_score > best_score
                    or (
                        candidate_score == best_score
                        and candidate_end - 1 > best_end
                    )
                ):
                    best_score = candidate_score
                    best_end = candidate_end - 1
        route[idx] = best_end
        scores[idx] = best_score


@export("mjb_route")
def mjb_route(
    chars_addr: Int,
    n: Int,
    edge_chars_addr: Int,
    edge_offsets_addr: Int,
    edge_children_addr: Int,
    node_weights_addr: Int,
    root_children_addr: Int,
    max_word_length: Int,
    unknown_weight: Float64,
    route_addr: Int,
    scores_addr: Int,
) abi("C"):
    var chars = U32Ptr(unsafe_from_address=chars_addr)
    var edge_chars = U32Ptr(unsafe_from_address=edge_chars_addr)
    var edge_offsets = IPtr(unsafe_from_address=edge_offsets_addr)
    var edge_children = IPtr(unsafe_from_address=edge_children_addr)
    var node_weights = FPtr(unsafe_from_address=node_weights_addr)
    var root_children = IPtr(unsafe_from_address=root_children_addr)
    var route = IPtr(unsafe_from_address=route_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    route_range(
        chars,
        0,
        n,
        edge_chars,
        edge_offsets,
        edge_children,
        node_weights,
        root_children,
        max_word_length,
        unknown_weight,
        route,
        scores,
    )


@export("mjb_route_batch")
def mjb_route_batch(
    chars_addr: Int,
    n: Int,
    starts_addr: Int,
    ends_addr: Int,
    segment_count: Int,
    edge_chars_addr: Int,
    edge_offsets_addr: Int,
    edge_children_addr: Int,
    node_weights_addr: Int,
    root_children_addr: Int,
    max_word_length: Int,
    unknown_weight: Float64,
    route_addr: Int,
    scores_addr: Int,
) abi("C"):
    var chars = U32Ptr(unsafe_from_address=chars_addr)
    var starts = IPtr(unsafe_from_address=starts_addr)
    var ends = IPtr(unsafe_from_address=ends_addr)
    var edge_chars = U32Ptr(unsafe_from_address=edge_chars_addr)
    var edge_offsets = IPtr(unsafe_from_address=edge_offsets_addr)
    var edge_children = IPtr(unsafe_from_address=edge_children_addr)
    var node_weights = FPtr(unsafe_from_address=node_weights_addr)
    var root_children = IPtr(unsafe_from_address=root_children_addr)
    var route = IPtr(unsafe_from_address=route_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    var workers = min(
        min(segment_count, num_physical_cores()), MAX_WORKERS
    )

    @parameter
    def process_worker(worker: Int):
        var first = segment_count * worker // workers
        var last = segment_count * (worker + 1) // workers
        for segment in range(first, last):
            route_range(
                chars,
                Int(starts[segment]),
                Int(ends[segment]),
                edge_chars,
                edge_offsets,
                edge_children,
                node_weights,
                root_children,
                max_word_length,
                unknown_weight,
                route,
                scores,
            )

    if n >= PARALLEL_THRESHOLD and workers > 1:
        parallelize[process_worker](workers, workers)
    else:
        for segment in range(segment_count):
            route_range(
                chars,
                Int(starts[segment]),
                Int(ends[segment]),
                edge_chars,
                edge_offsets,
                edge_children,
                node_weights,
                root_children,
                max_word_length,
                unknown_weight,
                route,
                scores,
            )
