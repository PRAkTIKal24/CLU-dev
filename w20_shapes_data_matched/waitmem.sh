#!/bin/bash
# waitmem.sh <GB>: block until at least <GB> of physical memory is free
until [ "$(powershell -NoProfile -c '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)')" -ge "$1" ]; do sleep 60; done
