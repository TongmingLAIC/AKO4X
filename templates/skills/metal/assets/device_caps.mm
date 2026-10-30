// Read device / optional compiled-pipeline properties. Never submit GPU commands.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::string source_path, function_name, language = "2.3";
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--help") {
                    std::cout << "metal-device-caps [--source FILE --function NAME --language VERSION]\n"
                                 "Reads device/pipeline capabilities; submits no GPU commands.\n";
                    return 0;
                }
                if (++i >= argc) throw std::runtime_error("missing option value");
                if (arg == "--source") source_path = argv[i];
                else if (arg == "--function") function_name = argv[i];
                else if (arg == "--language") language = argv[i];
                else throw std::runtime_error("unknown option: " + arg);
            }
            if (source_path.empty() != function_name.empty())
                throw std::runtime_error("--source and --function must be supplied together");
            std::map<std::string, MTLLanguageVersion> versions = {{"2.3", MTLLanguageVersion2_3}};
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
            if (@available(macOS 12.0, *)) versions.emplace("2.4", MTLLanguageVersion2_4);
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
            if (@available(macOS 13.0, *)) versions.emplace("3.0", MTLLanguageVersion3_0);
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
            if (@available(macOS 14.0, *)) versions.emplace("3.1", MTLLanguageVersion3_1);
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) versions.emplace("3.2", MTLLanguageVersion3_2);
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
            if (@available(macOS 26.0, *)) versions.emplace("4.0", MTLLanguageVersion4_0);
#endif
            if (!versions.count(language))
                throw std::runtime_error("language unavailable in this probe's SDK/OS mapping; check the target SDK's API");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no Metal device");
            NSMutableArray* families = [NSMutableArray array];
            if ([device supportsFamily:MTLGPUFamilyApple7]) [families addObject:@"Apple7"];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
            if ([device supportsFamily:MTLGPUFamilyApple8]) [families addObject:@"Apple8"];
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
            if ([device supportsFamily:MTLGPUFamilyApple9]) [families addObject:@"Apple9"];
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
            if ([device supportsFamily:MTLGPUFamilyApple10]) [families addObject:@"Apple10"];
#endif
            NSMutableArray* counter_sets = [NSMutableArray array];
            for (id<MTLCounterSet> set in device.counterSets) [counter_sets addObject:set.name];
            NSMutableDictionary* report = [@{
                @"device": device.name, @"registry_id": @(device.registryID),
                @"has_unified_memory": @(device.hasUnifiedMemory), @"supported_apple_families": families,
                @"device_max_threadgroup_memory_bytes": @(device.maxThreadgroupMemoryLength),
                @"device_max_threads_dimensions": @[@(device.maxThreadsPerThreadgroup.width),
                    @(device.maxThreadsPerThreadgroup.height), @(device.maxThreadsPerThreadgroup.depth)],
                @"max_buffer_bytes": @(device.maxBufferLength), @"counter_sets": counter_sets,
                @"dispatch_counter_sampling": @([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
                @"gpu_work_submitted": @NO
            } mutableCopy];
            if (!source_path.empty()) {
                NSError* error = nil;
                NSString* source = [NSString stringWithContentsOfFile:@(source_path.c_str())
                                                           encoding:NSUTF8StringEncoding error:&error];
                if (!source) throw std::runtime_error([[error description] UTF8String]);
                MTLCompileOptions* options = [MTLCompileOptions new];
                options.languageVersion = versions.at(language);
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
                if (@available(macOS 15.0, *)) {
                    options.mathMode = MTLMathModeSafe;
                    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
                } else
#endif
                {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                }
                id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
                if (!library) throw std::runtime_error([[error description] UTF8String]);
                id<MTLFunction> function = [library newFunctionWithName:@(function_name.c_str())];
                if (!function) throw std::runtime_error("function missing (specialization may be required)");
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error description] UTF8String]);
                report[@"pipeline"] = @{
                    @"function": @(function_name.c_str()), @"language": @(language.c_str()),
                    @"thread_execution_width": @(pipeline.threadExecutionWidth),
                    @"max_total_threads_per_threadgroup": @(pipeline.maxTotalThreadsPerThreadgroup),
                    @"static_threadgroup_memory_bytes": @(pipeline.staticThreadgroupMemoryLength)
                };
            }
            NSError* error = nil;
            NSData* json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            if (!json) throw std::runtime_error("JSON encoding failed");
            std::cout.write(static_cast<const char*>(json.bytes), json.length);
            std::cout << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
