file(REMOVE_RECURSE
  "../bin/hellocuda.pdb"
  "../bin/hellocuda"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/hellocuda.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
