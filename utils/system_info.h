#ifndef SYSTEMINFO_H
#define SYSTEMINFO_H

#include <string>

enum class PerformanceTier {
    LOW,
    MEDIUM,
    HIGH
};

class SystemInfo {
public:
    static int getRAM_GB();
    static std::string getCPUName();
    static PerformanceTier getPerformanceTier();
};

#endif