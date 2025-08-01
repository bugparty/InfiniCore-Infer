#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Runtime resources allocated per device. The structure collects
// handles, streams and weight tensors needed during inference on a
// single accelerator.
struct DeviceResource {
    // Device backend used for inference (CPU/GPU/other)
    infiniDevice_t device;
    // Physical id of the device
    int device_id;
    // InfiniCore operator handle for launching kernels
    infiniopHandle_t handle;
    // Model weights residing on this device
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv,
        w_attn_out, w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Execution stream associated with the device
    infinirtStream_t stream;
    // Optional NCCL style communicator for multi-device inference
    infinicclComm_t comm;

    // Pool for temporary buffers and workspace on this device
    std::shared_ptr<MemoryPool> memory_pool;
};

// Synchronisation primitives for the worker thread serving one device
struct InferState {
    std::mutex mtx;
    // Notifies when resources have been loaded
    std::condition_variable cv_load;
    // Signalled to start a batch
    std::condition_variable cv_start;
    // Signalled after batch is finished
    std::condition_variable cv_done;
    bool loaded = false;   // resources ready
    bool proceed = false;  // worker should process the request
    bool exit_flag = false; // thread termination flag
};

// Runtime request passed to all device threads each iteration
struct InferRequest {
    // Flattened input tokens for the entire batch
    const uint32_t *tokens;
    uint32_t ntok;
    // Length of each request in the batch
    const uint32_t *req_lens;
    uint32_t nreq;
    // Starting position for each request
    const uint32_t *req_pos;
    // Pointer to per-request KV caches
    struct KVCache **kv_caches;
    // Sampling parameters
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    // Output token ids written by device 0
    uint32_t *output;
};

// High level model wrapper owning worker threads for each device
struct JiugeModel {
    // Static model description
    JiugeMeta meta;
    // Device type (CPU/GPU/etc)
    infiniDevice_t device;
    // Physical device ids
    std::vector<int> dev_ids;
    // Resources for each device
    std::vector<DeviceResource> dev_resources;
    // Worker thread state for each device
    std::vector<InferState> states;
    // Worker threads
    std::vector<std::thread> threads;
    // Current inference request shared between workers
    InferRequest req;

    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

// Per-request key/value cache used by attention layers.
// Dimension order: [device][layer]
struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif
