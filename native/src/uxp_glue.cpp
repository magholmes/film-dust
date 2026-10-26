// Photoshop UXP hybrid addon binding (filmdust.uxpaddon). JS: require("filmdust.uxpaddon").
// The addon API mirrors Node-API (uxp-sdk/api/UxpAddonShared.h); calls arrive on UXP's scripting
// thread, so the multi-threaded filters may block it while they run.
#include <cstddef>
#include <cstring>

#include "../uxp-sdk/utilities/UxpAddon.h"
#include "filters.h"

namespace host {
using Env = addon_env;
using Val = addon_value;
using Info = addon_callback_info;

inline bool get_args(Env env, Info info, size_t* argc, Val* argv) {
    return UxpAddonApis.uxp_addon_get_cb_info(env, info, argc, argv, nullptr, nullptr) == addon_ok;
}
inline bool arraybuffer(Env env, Val v, void** data, size_t* len) {
    return UxpAddonApis.uxp_addon_get_arraybuffer_info(env, v, data, len) == addon_ok;
}
inline bool get_i32(Env env, Val v, int32_t* r) { return UxpAddonApis.uxp_addon_get_value_int32(env, v, r) == addon_ok; }
inline bool get_f64(Env env, Val v, double* r) { return UxpAddonApis.uxp_addon_get_value_double(env, v, r) == addon_ok; }
// (a host that left any of these out reads as "no such field": that call fails cleanly)
inline bool get_named(Env env, Val obj, const char* name, Val* r) {
    return UxpAddonApis.uxp_addon_get_named_property && UxpAddonApis.uxp_addon_get_named_property(env, obj, name, r) == addon_ok;
}
inline bool get_element(Env env, Val arr, uint32_t i, Val* r) {
    bool is = false;
    return UxpAddonApis.uxp_addon_is_array && UxpAddonApis.uxp_addon_get_element &&
           UxpAddonApis.uxp_addon_is_array(env, arr, &is) == addon_ok && is && UxpAddonApis.uxp_addon_get_element(env, arr, i, r) == addon_ok;
}
inline Val new_arraybuffer(Env env, size_t bytes, void** data) {
    Val r = nullptr;
    return UxpAddonApis.uxp_addon_create_arraybuffer(env, bytes, data, &r) == addon_ok ? r : nullptr;
}
inline Val number(Env env, double x) {
    Val r = nullptr;
    UxpAddonApis.uxp_addon_create_double(env, x, &r);
    return r;
}
inline void throw_error(Env env, const char* msg) { UxpAddonApis.uxp_addon_throw_error(env, nullptr, msg); }
}  // namespace host

#include "glue.inc"

static addon_value init(addon_env env, addon_value exports, const addon_apis& apis) {
    for (const auto& e : FD_FUNCTIONS) {
        addon_value fn = nullptr;
        apis.uxp_addon_create_function(env, e.name, std::strlen(e.name), e.fn, nullptr, &fn);
        apis.uxp_addon_set_named_property(env, exports, e.name, fn);
    }
    return exports;
}

UXP_ADDON_INIT(init)

// UXP unloads the addon: stop the worker threads before the DLL's code goes away
static void terminate(addon_env) { fd::shutdown_pool(); }
UXP_ADDON_TERMINATE(terminate)
