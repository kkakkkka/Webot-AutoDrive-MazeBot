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
CMAKE_SOURCE_DIR = C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4

# Include any dependencies generated for this target.
include CMakeFiles/controller4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/controller4.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/controller4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/controller4.dir/flags.make

CMakeFiles/controller4.dir/controller4.cpp.obj: CMakeFiles/controller4.dir/flags.make
CMakeFiles/controller4.dir/controller4.cpp.obj: CMakeFiles/controller4.dir/includes_CXX.rsp
CMakeFiles/controller4.dir/controller4.cpp.obj: controller4.cpp
CMakeFiles/controller4.dir/controller4.cpp.obj: CMakeFiles/controller4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/controller4.dir/controller4.cpp.obj"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/controller4.dir/controller4.cpp.obj -MF CMakeFiles\controller4.dir\controller4.cpp.obj.d -o CMakeFiles\controller4.dir\controller4.cpp.obj -c C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\controller4.cpp

CMakeFiles/controller4.dir/controller4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/controller4.dir/controller4.cpp.i"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\controller4.cpp > CMakeFiles\controller4.dir\controller4.cpp.i

CMakeFiles/controller4.dir/controller4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/controller4.dir/controller4.cpp.s"
	C:\Users\xu\Desktop\2021c\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\controller4.cpp -o CMakeFiles\controller4.dir\controller4.cpp.s

# Object files for target controller4
controller4_OBJECTS = \
"CMakeFiles/controller4.dir/controller4.cpp.obj"

# External object files for target controller4
controller4_EXTERNAL_OBJECTS =

controller4.exe: CMakeFiles/controller4.dir/controller4.cpp.obj
controller4.exe: CMakeFiles/controller4.dir/build.make
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_dnn348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_highgui348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_ml348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_objdetect348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_shape348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_stitching348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_superres348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_videostab348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_calib3d348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_features2d348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_flann348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_photo348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_video348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_videoio348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_imgcodecs348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_imgproc348.dll.a
controller4.exe: C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64/x64/mingw/lib/libopencv_core348.dll.a
controller4.exe: CMakeFiles/controller4.dir/linklibs.rsp
controller4.exe: CMakeFiles/controller4.dir/objects1.rsp
controller4.exe: CMakeFiles/controller4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable controller4.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\controller4.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/controller4.dir/build: controller4.exe
.PHONY : CMakeFiles/controller4.dir/build

CMakeFiles/controller4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\controller4.dir\cmake_clean.cmake
.PHONY : CMakeFiles/controller4.dir/clean

CMakeFiles/controller4.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4 C:\Users\xu\Desktop\vsc_file\robotf\FinalProject\test\mapping\controllers\controller4\CMakeFiles\controller4.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/controller4.dir/depend
