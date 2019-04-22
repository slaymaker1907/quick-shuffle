# Inspired by https://gist.github.com/mauriciopoppe/de8908f67923091982c8c8136a063ea6
CXX = nvcc

# Taken from https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

BUILD_PATH ?= build

# code lists #
SOURCES = $(call rwildcard, src/, *.cpp)
CU_SOURCES = $(call rwildcard, src/, *.cu)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_PATH)/%.o)
CU_OBJECTS = $(CU_SOURCES:src/%.cu=$(BUILD_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:%.o=%.d)
CU_DEPS = $(CU_OBJECTS:%.o=%.d)

# flags #
# CXX_FLAGS = -std=c++11 -Wall -Wextra -g
INCLUDES = -I include/ -I ../cub-1.8.0/cub
CXX_FLAGS =--compiler-options=-Wall --Wreorder -O3 --std=c++14
# CXX_FLAGS =--compiler-options=-Wall --std=c++14 -lineinfo
# CXX_FLAGS =--compiler-options=-Wall --std=c++14 -G
# Space-separated pkg-config libraries used by this project

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS)) $(dir $(CU_OBJECTS))

.PHONY: clean
clean:
	@$(RM) -r build
	@$(RM) main

# checks the executable and symlinks to the output
.PHONY: all
all: dirs main

# Creation of the executable
main: $(OBJECTS) $(CU_OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(CXX_FLAGS) $(OBJECTS) $(CU_OBJECTS) -o $@

$(DEPS): $(BUILD_PATH)/%.d: src/%.cpp
	@echo "Generating dependency file for $<"
	$(CXX) $(INCLUDES) -M $< -o $@

$(CU_DEPS): $(BUILD_PATH)/%.d: src/%.cu
	@echo "Generating dependency file for $<"
	$(CXX) $(INCLUDES) -M $< -o $@

# Add dependency files, if they exist
-include $(DEPS)
-include $(CU_DEPS)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: src/%.cpp
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_PATH)/%.o: src/%.cu
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@
