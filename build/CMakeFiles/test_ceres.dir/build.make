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
CMAKE_SOURCE_DIR = /home/cyy/projects/test_ceres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cyy/projects/test_ceres/build

# Include any dependencies generated for this target.
include CMakeFiles/test_ceres.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_ceres.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_ceres.dir/flags.make

CMakeFiles/test_ceres.dir/src/main.cpp.o: CMakeFiles/test_ceres.dir/flags.make
CMakeFiles/test_ceres.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cyy/projects/test_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_ceres.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_ceres.dir/src/main.cpp.o -c /home/cyy/projects/test_ceres/src/main.cpp

CMakeFiles/test_ceres.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ceres.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cyy/projects/test_ceres/src/main.cpp > CMakeFiles/test_ceres.dir/src/main.cpp.i

CMakeFiles/test_ceres.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ceres.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cyy/projects/test_ceres/src/main.cpp -o CMakeFiles/test_ceres.dir/src/main.cpp.s

CMakeFiles/test_ceres.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/test_ceres.dir/src/main.cpp.o.requires

CMakeFiles/test_ceres.dir/src/main.cpp.o.provides: CMakeFiles/test_ceres.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_ceres.dir/build.make CMakeFiles/test_ceres.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/test_ceres.dir/src/main.cpp.o.provides

CMakeFiles/test_ceres.dir/src/main.cpp.o.provides.build: CMakeFiles/test_ceres.dir/src/main.cpp.o


# Object files for target test_ceres
test_ceres_OBJECTS = \
"CMakeFiles/test_ceres.dir/src/main.cpp.o"

# External object files for target test_ceres
test_ceres_EXTERNAL_OBJECTS =

devel/lib/test_ceres/test_ceres: CMakeFiles/test_ceres.dir/src/main.cpp.o
devel/lib/test_ceres/test_ceres: CMakeFiles/test_ceres.dir/build.make
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libroscpp.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/librosconsole.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libroscpp_serialization.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/librostime.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libxmlrpcpp.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libcpp_common.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libroslib.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/librospack.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libpython2.7.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libtinyxml.so
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_stitching3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_superres3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_videostab3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_aruco3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_bgsegm3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_bioinspired3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_ccalib3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_cvv3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_dpm3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_face3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_fuzzy3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_hdf3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_img_hash3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_line_descriptor3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_optflow3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_reg3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_rgbd3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_saliency3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_stereo3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_structured_light3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_surface_matching3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_tracking3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_ximgproc3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_xobjdetect3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_xphoto3.so.3.3.1
devel/lib/test_ceres/test_ceres: /usr/local/lib/libceres.a
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_shape3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_photo3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_calib3d3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_viz3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_video3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_datasets3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_plot3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_text3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_dnn3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_features2d3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_flann3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_highgui3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_ml3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_videoio3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_imgcodecs3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_objdetect3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_imgproc3.so.3.3.1
devel/lib/test_ceres/test_ceres: /opt/ros/kinetic/lib/libopencv_core3.so.3.3.1
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libglog.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libgflags.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libspqr.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libtbb.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcholmod.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libccolamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcolamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/liblapack.so
devel/lib/test_ceres/test_ceres: /usr/lib/libf77blas.so
devel/lib/test_ceres/test_ceres: /usr/lib/libatlas.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/librt.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libspqr.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libtbb.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcholmod.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libccolamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcolamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libamd.so
devel/lib/test_ceres/test_ceres: /usr/lib/liblapack.so
devel/lib/test_ceres/test_ceres: /usr/lib/libf77blas.so
devel/lib/test_ceres/test_ceres: /usr/lib/libatlas.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/librt.so
devel/lib/test_ceres/test_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
devel/lib/test_ceres/test_ceres: CMakeFiles/test_ceres.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cyy/projects/test_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable devel/lib/test_ceres/test_ceres"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ceres.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_ceres.dir/build: devel/lib/test_ceres/test_ceres

.PHONY : CMakeFiles/test_ceres.dir/build

CMakeFiles/test_ceres.dir/requires: CMakeFiles/test_ceres.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/test_ceres.dir/requires

CMakeFiles/test_ceres.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_ceres.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_ceres.dir/clean

CMakeFiles/test_ceres.dir/depend:
	cd /home/cyy/projects/test_ceres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cyy/projects/test_ceres /home/cyy/projects/test_ceres /home/cyy/projects/test_ceres/build /home/cyy/projects/test_ceres/build /home/cyy/projects/test_ceres/build/CMakeFiles/test_ceres.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_ceres.dir/depend
