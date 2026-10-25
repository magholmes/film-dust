// Photoshop UXP hybrid addon binding (filmdust.uxpaddon). JS: require("filmdust.uxpaddon").
// The addon API mirrors Node-API (uxp-sdk/api/UxpAddonShared.h); calls arrive on UXP's scripting
// thread, so the multi-threaded filters may block it while they run.
#include <cstddef>
#include <cstring>

#include "../uxp-sdk/utilities/UxpAddon.h"
#include "filters.h"

#define ENV addon_env
#define VAL addon_value
#define CBINFO addon_callback_info
#define ARGS(env, info, n)        \
    size_t argc = n;              \
    addon_value argv[n] = {};     \
    UxpAddonApis.uxp_addon_get_cb_info(env, info, &argc, argv, nullptr, nullptr)

template <typename T>
static T* ab_ptr(addon_env env, addon_value v, size_t off, size_t n) {
    void* data = nullptr;
    size_t len = 0;
    if (UxpAddonApis.uxp_addon_get_arraybuffer_info(env, v, &data, &len) != addon_ok || !data) return nullptr;
    if (off + n * sizeof(T) > len) return nullptr;
    return (T*)((char*)data + off);
}
#define AB(env, v, off, T, n) ab_ptr<T>(env, v, off, n)

static int i32(addon_env env, addon_value v) {
    int32_t r = 0;
    UxpAddonApis.uxp_addon_get_value_int32(env, v, &r);
    return r;
}
static double f64(addon_env env, addon_value v) {
    double r = 0;
    UxpAddonApis.uxp_addon_get_value_double(env, v, &r);
    return r;
}
#define I32(env, v) i32(env, v)
#define F64(env, v) f64(env, v)

static addon_value new_ab(addon_env env, size_t bytes, void** ptr) {
    addon_value r = nullptr;
    UxpAddonApis.uxp_addon_create_arraybuffer(env, bytes, ptr, &r);
    return r;
}
#define NEW_AB(env, bytes, pptr) new_ab(env, bytes, pptr)

static addon_value num(addon_env env, double x) {
    addon_value r = nullptr;
    UxpAddonApis.uxp_addon_create_double(env, x, &r);
    return r;
}
#define NUM(env, x) num(env, x)
#define THROW(env, msg) UxpAddonApis.uxp_addon_throw_error(env, nullptr, msg)

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
