# CMake generated Testfile for 
# Source directory: /home/guest-intern/git/jan-2/flappie-x-cpp-gpu
# Build directory: /home/guest-intern/git/jan-2/flappie-x-cpp-gpu/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unittest "/home/guest-intern/git/jan-2/flappie-x-cpp-gpu/build/flappie_unittest")
set_tests_properties(unittest PROPERTIES  WORKING_DIRECTORY "/home/guest-intern/git/jan-2/flappie-x-cpp-gpu/src/test/")
add_test(test_call "flappie" "/home/guest-intern/git/jan-2/flappie-x-cpp-gpu/reads")
add_test(test_licence "flappie" "--licence")
add_test(test_license "flappie" "--license")
add_test(test_help "flappie" "--help")
add_test(test_version "flappie" "--version")
