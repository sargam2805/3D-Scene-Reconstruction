# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/voldemort/Desktop/temp/proj/fastfusion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/voldemort/Desktop/temp/proj/fastfusion

# Include any dependencies generated for this target.
include src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/depend.make

# Include the progress variables for this target.
include src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/flags.make

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/flags.make
src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o: src/fusion/geometryfusion_mipmap_cpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/voldemort/Desktop/temp/proj/fastfusion/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o -c /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/geometryfusion_mipmap_cpu.cpp

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.i"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/geometryfusion_mipmap_cpu.cpp > CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.i

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.s"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/geometryfusion_mipmap_cpu.cpp -o CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.s

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.requires:

.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.requires

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.provides: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.requires
	$(MAKE) -f src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/build.make src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.provides.build
.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.provides

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.provides.build: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o


src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/flags.make
src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o: src/fusion/loopclosure.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/voldemort/Desktop/temp/proj/fastfusion/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o -c /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/loopclosure.cpp

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.i"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/loopclosure.cpp > CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.i

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.s"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/loopclosure.cpp -o CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.s

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.requires:

.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.requires

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.provides: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.requires
	$(MAKE) -f src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/build.make src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.provides.build
.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.provides

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.provides.build: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o


# Object files for target geometryfusion_mipmap_cpu
geometryfusion_mipmap_cpu_OBJECTS = \
"CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o" \
"CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o"

# External object files for target geometryfusion_mipmap_cpu
geometryfusion_mipmap_cpu_EXTERNAL_OBJECTS =

lib/libgeometryfusion_mipmap_cpu.a: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o
lib/libgeometryfusion_mipmap_cpu.a: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o
lib/libgeometryfusion_mipmap_cpu.a: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/build.make
lib/libgeometryfusion_mipmap_cpu.a: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/voldemort/Desktop/temp/proj/fastfusion/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library ../../lib/libgeometryfusion_mipmap_cpu.a"
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && $(CMAKE_COMMAND) -P CMakeFiles/geometryfusion_mipmap_cpu.dir/cmake_clean_target.cmake
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/geometryfusion_mipmap_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/build: lib/libgeometryfusion_mipmap_cpu.a

.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/build

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/requires: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/geometryfusion_mipmap_cpu.cpp.o.requires
src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/requires: src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/loopclosure.cpp.o.requires

.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/requires

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/clean:
	cd /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion && $(CMAKE_COMMAND) -P CMakeFiles/geometryfusion_mipmap_cpu.dir/cmake_clean.cmake
.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/clean

src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/depend:
	cd /home/voldemort/Desktop/temp/proj/fastfusion && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/voldemort/Desktop/temp/proj/fastfusion /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion /home/voldemort/Desktop/temp/proj/fastfusion /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion /home/voldemort/Desktop/temp/proj/fastfusion/src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/fusion/CMakeFiles/geometryfusion_mipmap_cpu.dir/depend

