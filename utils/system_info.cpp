// 1.0
#include "system_info.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

int SystemInfo::getRAM_GB()
{

#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (int)(status.ullTotalPhys / (1024LL * 1024LL * 1024LL));

#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t size = 0;
    size_t len = sizeof(size);
    sysctl(mib, 2, &size, &len, NULL, 0);
    return (int)(size / (1024LL * 1024LL * 1024LL));

#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long long total = (long long)pages * (long long)page_size;
    return (int)(total / (1024LL * 1024LL * 1024LL));

#else
    return 8;
#endif
}

std::string SystemInfo::getCPUName()
{
    return "GenericCPU";
}

PerformanceTier SystemInfo::getPerformanceTier()
{
    int ram = getRAM_GB();

    if(ram <= 16) { return PerformanceTier::LOW; }
    if(ram <= 32) { return PerformanceTier::MEDIUM; }

    return PerformanceTier::HIGH;
}