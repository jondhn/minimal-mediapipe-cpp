@echo off

set "set_compiler="
IF not "%1%"=="" (
IF "%1%"=="17" (set "set_compiler=-s compiler.version=17")
)

erase /Q .\build\conan\*.*
rem --build missing

@REM debug build
conan install ./conan/conanfile.txt --build missing ^
 --profile:build ./conan/conan_profile_debug.txt ^
 --profile:host ./conan/conan_profile_debug.txt  ^
 %set_compiler% ^
 --output-folder=./build/conan ^
 -g CMakeDeps || goto err

@REM release build
conan install ./conan/conanfile.txt --build missing ^
 --profile:build ./conan/conan_profile_release.txt ^
 --profile:host ./conan/conan_profile_release.txt ^
 %set_compiler% ^
 --output-folder=./build/conan ^
 -g CMakeDeps || goto err



exit /B 0

:err
exit /B 1
