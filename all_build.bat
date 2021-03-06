@echo off
pushd "%~dp0"

echo -----Editing props-----
powershell -executionpolicy bypass -file edit_props.ps1

echo #### nuget packages install
mkdir grpc\vsprojects\packages & cd grpc\vsprojects\packages
powershell -executionpolicy bypass -Command Invoke-WebRequest https://dist.nuget.org/win-x86-commandline/latest/nuget.exe -OutFile "%cd%\nuget.exe"
nuget.exe install ..\vcxproj\grpc\packages.config
cd ..\..\..\

echo ----Props editting done-----

@setlocal

@REM EDIT THIS SECTION ACCORDING TO YOUR ENV
if not defined DevEnvDir (
	call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
)
set path=%path%;C:\Program Files (x86)\CMake\bin
set path=%path%;C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE
set path=%path%;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
@REM EOF

echo -----Building grpc-----

set devenv=devenv

mkdir grpc\bin\zlib
mkdir grpc\bin\zlib\debug
mkdir grpc\bin\zlib\release

mkdir lib

cd grpc\third_party\zlib
mkdir build & cd build
mkdir debug & cd debug
cmake -Wno-error -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../../../../bin/zlib/debug ../..
nmake & nmake install

cd ..
mkdir release & cd release
cmake -Wno-error -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../../../bin/zlib/release ../..
nmake & nmake install

cd ../../../../bin/zlib/release
set PATH=%PATH%;%cd%\bin

popd
pushd "%~dp0"

cd grpc\third_party\protobuf\cmake

mkdir build & cd build
mkdir solution & cd solution
cmake -Wno-error -G "Visual Studio 14 2015 Win64" -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB=ON ../..
"%devenv%" protobuf.sln /build "Debug|x64" /project ALL_BUILD
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\Debug ..\..\..\..\..\bin\protobuf\debug

"%devenv%" protobuf.sln /build "Release|x64" /project ALL_BUILD
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\Release ..\..\..\..\..\bin\protobuf\release

cd ..\..\..\..\..\vsprojects
"%devenv%" grpc_protoc_plugins.sln /build "Release|x64"
if not %ERRORLEVEL% == 0 goto Finish
robocopy .\x64\Release\ ..\bin\grpc_protoc_plugins\ /XF *.lib *.iobj *.ipdb
"%devenv%" grpc_protoc_plugins.sln /clean "Release|x64"

"%devenv%" grpc.sln /clean "Debug"
"%devenv%" grpc.sln /clean "Release"
"%devenv%" grpc.sln /build "Debug|x64" /project grpc++
"%devenv%" grpc.sln /build "Debug|x64" /project grpc++_unsecure
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\x64\Debug ..\bin\grpc\debug

"%devenv%" grpc.sln /build "Release|x64" /project grpc++
"%devenv%" grpc.sln /build "Release|x64" /project grpc++_unsecure
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\x64\Release ..\bin\grpc\release /XF *grpc_cpp_plugin*

"%devenv%" grpc.sln /clean "Debug"
"%devenv%" grpc.sln /clean "Release"
"%devenv%" grpc.sln /build "Debug-DLL|x64" /project grpc++
"%devenv%" grpc.sln /build "Debug-DLL|x64" /project grpc++_unsecure
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\x64\Debug-DLL ..\bin\grpc\debug_dll

"%devenv%" grpc.sln /build "Release-DLL|x64" /project grpc++
"%devenv%" grpc.sln /build "Release-DLL|x64" /project grpc++_unsecure
if not %ERRORLEVEL% == 0 goto Finish
robocopy /mir .\x64\Release-DLL ..\bin\grpc\release_dll /XF *grpc_cpp_plugin*

robocopy /mir .\packages\grpc.dependencies.openssl.1.0.204.1\build\native\lib\v140\x64\Release\static ..\bin\dependencies

echo -----grpc building done-----

cd ../../

echo -----Building pb files-----

grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\acquisition.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\analysis_configuration.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\data.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\instance.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\keystore.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\log.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\manager.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\minion_device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\promethion_device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\protocol.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\rpc_options.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --grpc_out=include\minknow\rpc\ --plugin=protoc-gen-grpc=grpc\bin\grpc_protoc_plugins\grpc_cpp_plugin.exe include\minknow\rpc\statistics.proto


grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\acquisition.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\analysis_configuration.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\data.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\instance.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\keystore.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\log.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\manager.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\minion_device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\promethion_device.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\protocol.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\rpc_options.proto
grpc\bin\protobuf\release\protoc.exe -I include\minknow\rpc\ -I grpc\third_party\protobuf\src\ --cpp_out=include\minknow\rpc\ include\minknow\rpc\statistics.proto

echo -----pb building done-----

echo -----building pthreads-----

cd pthread-win32

"%devenv%" pthread.2015.sln /build "Release|x64" /project pthread_dll
if not %ERRORLEVEL% == 0 goto Finish
"%devenv%" pthread.2015.sln /build "Release|x64" /project pthread_lib
if not %ERRORLEVEL% == 0 goto Finish
robocopy .\bin\x64_MSVC2015.Release\ ..\lib\ /XF *.exp *.map

echo -----pthread building done-----


:Finish

popd
endlocal
pause
