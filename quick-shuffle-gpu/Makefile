# Inspired by https://gist.github.com/mauriciopoppe/de8908f67923091982c8c8136a063ea6
CXX ?= g++
CCUDA ?= /uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/bin/nvcc


# Taken from https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

# code lists #
CPP_SOURCES = $(call rwildcard, src/, *.cpp)
CUDA_SOURCES = $(call rwildcard, src/, *.cu)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
CPP_OBJECTS = $(CPP_SOURCES:src/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
CUDA_OBJECTS = $(CPP_SOURCES:src/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)
# Set the dependency files that will be used to add header dependencies
CPP_DEPS = $(CPP_OBJECTS:.o=.d)
CUDA_DEPS = $(CUDA_OBJECTS:.o=.d)
DEPS = $(CPP_DEPS) $(CUDA_DEPS)

# flags #
CXX_FLAGS = -std=c++11 -Wall -Wextra -g
INCLUDES = -I include/ -I /usr/local/include -I /uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include
CUDA_FLAGS =--compiler-options=-Wall --Wreorder
# Space-separated pkg-config libraries used by this project

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))

.PHONY: clean
clean:
	@$(RM) -r build
	@$(RM) main

# checks the executable and symlinks to the output
.PHONY: all
all: main

# Creation of the executable
main: $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS) -o $@

# Add dependency files, if they exist
-include $(DEPS)

$(CPP_DEPS): build/%.d: src/%.cpp
	@echo "Compiling dependency: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

$(CUDA_DEPS): build/%.d: src/%.cu
	@echo "Compiling dependency: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

build/%.o: src/%.cpp
	@echo "Compiling: $< -> $@"
	$(CCUDA) $(CUDA_FLAGS) $(INCLUDES) $< -o

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@
