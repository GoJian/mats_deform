# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:
.PHONY : .NOTPARALLEL

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
CMAKE_SOURCE_DIR = /home/GoJian/Programs/FEM/project/mats_deform

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/GoJian/Programs/FEM/project/mats_deform

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/GoJian/Programs/FEM/project/mats_deform/CMakeFiles /home/GoJian/Programs/FEM/project/mats_deform/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/GoJian/Programs/FEM/project/mats_deform/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named debug

# Build rule for target.
debug: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 debug
.PHONY : debug

# fast build rule for target.
debug/fast:
	$(MAKE) -f CMakeFiles/debug.dir/build.make CMakeFiles/debug.dir/build
.PHONY : debug/fast

#=============================================================================
# Target rules for targets named distclean

# Build rule for target.
distclean: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 distclean
.PHONY : distclean

# fast build rule for target.
distclean/fast:
	$(MAKE) -f CMakeFiles/distclean.dir/build.make CMakeFiles/distclean.dir/build
.PHONY : distclean/fast

#=============================================================================
# Target rules for targets named info

# Build rule for target.
info: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 info
.PHONY : info

# fast build rule for target.
info/fast:
	$(MAKE) -f CMakeFiles/info.dir/build.make CMakeFiles/info.dir/build
.PHONY : info/fast

#=============================================================================
# Target rules for targets named mats_deform

# Build rule for target.
mats_deform: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mats_deform
.PHONY : mats_deform

# fast build rule for target.
mats_deform/fast:
	$(MAKE) -f CMakeFiles/mats_deform.dir/build.make CMakeFiles/mats_deform.dir/build
.PHONY : mats_deform/fast

#=============================================================================
# Target rules for targets named release

# Build rule for target.
release: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 release
.PHONY : release

# fast build rule for target.
release/fast:
	$(MAKE) -f CMakeFiles/release.dir/build.make CMakeFiles/release.dir/build
.PHONY : release/fast

#=============================================================================
# Target rules for targets named run

# Build rule for target.
run: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 run
.PHONY : run

# fast build rule for target.
run/fast:
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/build
.PHONY : run/fast

#=============================================================================
# Target rules for targets named runclean

# Build rule for target.
runclean: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 runclean
.PHONY : runclean

# fast build rule for target.
runclean/fast:
	$(MAKE) -f CMakeFiles/runclean.dir/build.make CMakeFiles/runclean.dir/build
.PHONY : runclean/fast

#=============================================================================
# Target rules for targets named strip_comments

# Build rule for target.
strip_comments: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 strip_comments
.PHONY : strip_comments

# fast build rule for target.
strip_comments/fast:
	$(MAKE) -f CMakeFiles/strip_comments.dir/build.make CMakeFiles/strip_comments.dir/build
.PHONY : strip_comments/fast

mats_deform.o: mats_deform.cc.o
.PHONY : mats_deform.o

# target to build an object file
mats_deform.cc.o:
	$(MAKE) -f CMakeFiles/mats_deform.dir/build.make CMakeFiles/mats_deform.dir/mats_deform.cc.o
.PHONY : mats_deform.cc.o

mats_deform.i: mats_deform.cc.i
.PHONY : mats_deform.i

# target to preprocess a source file
mats_deform.cc.i:
	$(MAKE) -f CMakeFiles/mats_deform.dir/build.make CMakeFiles/mats_deform.dir/mats_deform.cc.i
.PHONY : mats_deform.cc.i

mats_deform.s: mats_deform.cc.s
.PHONY : mats_deform.s

# target to generate assembly for a file
mats_deform.cc.s:
	$(MAKE) -f CMakeFiles/mats_deform.dir/build.make CMakeFiles/mats_deform.dir/mats_deform.cc.s
.PHONY : mats_deform.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... debug"
	@echo "... distclean"
	@echo "... edit_cache"
	@echo "... info"
	@echo "... mats_deform"
	@echo "... rebuild_cache"
	@echo "... release"
	@echo "... run"
	@echo "... runclean"
	@echo "... strip_comments"
	@echo "... mats_deform.o"
	@echo "... mats_deform.i"
	@echo "... mats_deform.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

