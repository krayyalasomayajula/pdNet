# Install script for directory: /mnt/harddisk/kalyan/primal-dual-networks/common/icgnn

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/kalyan/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so"
         RPATH "$ORIGIN/../lib:/home/kalyan/torch/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib" TYPE MODULE FILES "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/build/libicgnn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so"
         OLD_RPATH "/home/kalyan/torch/install/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/kalyan/torch/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lua/icgnn" TYPE FILE FILES
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNablaT.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgL2Norm.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgAddition.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgExpMul.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/init.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgAddConstants.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgResample.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgMask.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgGeneralizedNabla.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgCAddTable.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNabla.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNarrow.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold2.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNoise.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold3.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgGeneralizedNablaT.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so"
         RPATH "$ORIGIN/../lib:/home/kalyan/torch/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib" TYPE MODULE FILES "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/build/libicgnn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so"
         OLD_RPATH "/home/kalyan/torch/install/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/kalyan/torch/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lib/libicgnn.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/../../../../usr/local/lua/icgnn" TYPE FILE FILES
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNablaT.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgL2Norm.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgAddition.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgExpMul.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/init.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgAddConstants.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgResample.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgMask.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgGeneralizedNabla.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgCAddTable.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNabla.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNarrow.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold2.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgNoise.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgThreshold3.lua"
    "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/IcgGeneralizedNablaT.lua"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/mnt/harddisk/kalyan/primal-dual-networks/common/icgnn/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
