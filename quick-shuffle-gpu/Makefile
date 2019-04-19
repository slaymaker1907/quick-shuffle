# Inspired by https://gist.github.com/mauriciopoppe/de8908f67923091982c8c8136a063ea6
CXX = nvcc

# Taken from https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

BUILD_PATH ?= build

# code lists #
SOURCES = $(call rwildcard, src/, *.cpp)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:%.o=%.d)

# flags #
# CXX_FLAGS = -std=c++11 -Wall -Wextra -g
INCLUDES = -I include/
CXX_FLAGS =--compiler-options=-Wall --Wreorder -O3 --std=c++14
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
.PHONY: all dirs
all: main

# Creation of the executable
main: $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS) -o $@

$(DEPS): $(BUILD_PATH)/%.d: src/%.cpp
	@echo "Generating dependency file for $<"
	$(CXX) $(INCLUDES) -M $< -o $@

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: src/%.cpp
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

