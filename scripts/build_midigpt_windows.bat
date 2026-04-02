@echo off
setlocal

set "REPO_ROOT=%~dp0.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
set "NINJA_EXE=%REPO_ROOT%\.venv\Scripts\ninja.exe"
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
set "PYTHON_ROOT=C:\Users\micha\miniforge3"
set "TOOLCHAIN_FILE=%REPO_ROOT%\vendor\vcpkg\scripts\buildsystems\vcpkg.cmake"
set "TORCH_DIR=%REPO_ROOT%\.venv\Lib\site-packages\torch\share\cmake\Torch"
set "MIDI_GPT_DIR=%REPO_ROOT%\vendor\MIDI-GPT"

call "%VSDEVCMD%" -arch=x64 -host_arch=x64 || exit /b 1
cd /d "%MIDI_GPT_DIR%" || exit /b 1

if exist build-ninja rmdir /s /q build-ninja

set "VCPKG_TARGET_TRIPLET=x64-windows"
"%CMAKE_EXE%" -S . -B build-ninja -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_FILE%" ^
  -DPYTHON_EXECUTABLE="%PYTHON_EXE%" ^
  -DPython_EXECUTABLE="%PYTHON_EXE%" ^
  -DPython_ROOT_DIR="%PYTHON_ROOT%" ^
  -DPython_FIND_STRATEGY=LOCATION ^
  -DTorch_DIR="%TORCH_DIR%"
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build build-ninja
if errorlevel 1 exit /b 1

endlocal
