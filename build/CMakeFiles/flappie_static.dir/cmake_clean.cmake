file(REMOVE_RECURSE
  "libflappie.pdb"
  "libflappie.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/flappie_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
