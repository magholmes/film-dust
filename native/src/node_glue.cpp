// Node-API binding (filmdust.node): the same functions as the Photoshop addon, so the engine's
// native path can be tested and timed in node against the plain JS engine.
#define NAPI_VERSION 8
#include <node_api.h>

#include <cstddef>

#include "filters.h"

namespace host {
using Env = napi_env;
using Val = napi_value;
using Info = napi_callback_info;

inline bool get_args(Env env, Info info, size_t* argc, Val* argv) {
    return napi_get_cb_info(env, info, argc, argv, nullptr, nullptr) == napi_ok;
}
inline bool arraybuffer(Env env, Val v, void** data, size_t* len) {
    return napi_get_arraybuffer_info(env, v, data, len) == napi_ok;
}
inline bool get_i32(Env env, Val v, int32_t* r) { return napi_get_value_int32(env, v, r) == napi_ok; }
inline bool get_f64(Env env, Val v, double* r) { return napi_get_value_double(env, v, r) == napi_ok; }
inline bool get_named(Env env, Val obj, const char* name, Val* r) { return napi_get_named_property(env, obj, name, r) == napi_ok; }
inline bool get_element(Env env, Val arr, uint32_t i, Val* r) {
    bool is = false;
    return napi_is_array(env, arr, &is) == napi_ok && is && napi_get_element(env, arr, i, r) == napi_ok;
}
inline Val new_arraybuffer(Env env, size_t bytes, void** data) {
    Val r = nullptr;
    return napi_create_arraybuffer(env, bytes, data, &r) == napi_ok ? r : nullptr;
}
inline Val number(Env env, double x) {
    Val r = nullptr;
    napi_create_double(env, x, &r);
    return r;
}
inline void throw_error(Env env, const char* msg) { napi_throw_error(env, nullptr, msg); }
}  // namespace host

#include "glue.inc"

NAPI_MODULE_INIT() {
    for (const auto& e : FD_FUNCTIONS) {
        napi_value fn;
        napi_create_function(env, e.name, NAPI_AUTO_LENGTH, e.fn, nullptr, &fn);
        napi_set_named_property(env, exports, e.name, fn);
    }
    return exports;
}
