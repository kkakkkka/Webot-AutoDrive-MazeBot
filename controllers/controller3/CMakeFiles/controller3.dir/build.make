# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3

# Include any dependencies generated for this target.
include CMakeFiles/controller3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/controller3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/controller3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/controller3.dir/flags.make

CMakeFiles/controller3.dir/controller3.cpp.obj: CMakeFiles/controller3.dir/flags.make
CMakeFiles/controller3.dir/controller3.cpp.obj: CMakeFiles/controller3.dir/includes_CXX.rsp
CMakeFiles/controller3.dir/controller3.cpp.obj: controller3.cpp
CMakeFiles/controller3.dir/controller3.cpp.obj: CMakeFiles/controller3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/controller3.dir/controller3.cpp.obj"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/controller3.dir/controller3.cpp.obj -MF CMakeFiles\controller3.dir\controller3.cpp.obj.d -o CMakeFiles\controller3.dir\controller3.cpp.obj -c C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\controller3.cpp

CMakeFiles/controller3.dir/controller3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/controller3.dir/controller3.cpp.i"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\controller3.cpp > CMakeFiles\controller3.dir\controller3.cpp.i

CMakeFiles/controller3.dir/controller3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/controller3.dir/controller3.cpp.s"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\controller3.cpp -o CMakeFiles\controller3.dir\controller3.cpp.s

# Object files for target controller3
controller3_OBJECTS = \
"CMakeFiles/controller3.dir/controller3.cpp.obj"

# External object files for target controller3
controller3_EXTERNAL_OBJECTS =

controller3.exe: CMakeFiles/controller3.dir/controller3.cpp.obj
controller3.exe: CMakeFiles/controller3.dir/build.make
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_dnn348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_highgui348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_ml348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_objdetect348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_shape348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_stitching348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_superres348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_videostab348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_calib3d348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_features2d348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_flann348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_photo348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_video348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_videoio348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_imgcodecs348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_imgproc348.dll.a
controller3.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_core348.dll.a
controller3.exe: CMakeFiles/controller3.dir/linklibs.rsp
controller3.exe: CMakeFiles/controller3.dir/objects1.rsp
controller3.exe: CMakeFiles/controller3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable controller3.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\controller3.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/controller3.dir/build: controller3.exe
.PHONY : CMakeFiles/controller3.dir/build

CMakeFiles/controller3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\controller3.dir\cmake_clean.cmake
.PHONY : CMakeFiles/controller3.dir/clean

CMakeFiles/controller3.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller3\CMakeFiles\controller3.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/controller3.dir/depend
