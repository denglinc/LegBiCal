cmake_minimum_required(VERSION 3.18)
project(legbical_prime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pinocchio 2.6.20 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem serialization system)

set(PRIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/PRIME")
add_library(prime_crocoddyl SHARED
  "${PRIME_ROOT}/src/core/solver-base.cpp"
  "${PRIME_ROOT}/src/core/solvers/ddp.cpp"
  "${PRIME_ROOT}/src/core/solvers/fddp.cpp"
  "${PRIME_ROOT}/src/core/utils/exception.cpp"
  "${PRIME_ROOT}/src/core/utils/stop-watch.cpp"
  "${PRIME_ROOT}/src/contact_id/problem/contact-anitescu-id.cpp")
target_include_directories(prime_crocoddyl PUBLIC "${PRIME_ROOT}/include")
target_link_libraries(prime_crocoddyl PUBLIC
  pinocchio::pinocchio
  Boost::filesystem
  Boost::serialization
  Boost::system)
target_compile_definitions(
  prime_crocoddyl PUBLIC PINOCCHIO_ENABLE_COMPATIBILITY_WITH_VERSION_2)

add_executable(prime_fie native/prime_fie.cpp)
target_link_libraries(prime_fie PRIVATE prime_crocoddyl)

foreach(target prime_crocoddyl prime_fie)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-array-bounds
    -Wno-conversion
    -Wno-deprecated-declarations
    -Wno-reorder
    -Wno-sign-compare
    -Wno-unused-but-set-variable
    -Wno-unused-variable)
endforeach()
