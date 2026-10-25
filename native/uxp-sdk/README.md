# Adobe UXP hybrid plugin SDK

The Photoshop build of the native part (`filmdust.uxpaddon`) compiles against four files from Adobe's UXP hybrid
plugin SDK. They're Adobe's, so they aren't in this repo. Put them here like this:

```
native/uxp-sdk/api/UxpAddonShared.h
native/uxp-sdk/api/UxpAddonTypes.h
native/uxp-sdk/utilities/UxpAddon.cpp
native/uxp-sdk/utilities/UxpAddon.h
```

They come with Adobe's hybrid plugin samples for UXP. You only need them to build the Photoshop addon yourself:
the releases already have it built in, and the Node build used by the tests (`filmdust.node`) doesn't need them.
