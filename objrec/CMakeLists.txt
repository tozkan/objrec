project( objrec C CXX )

#check for boost
find_package(Boost 1.40 COMPONENTS program_options)

set(OBJREC_SRCS feature2d.cpp object_detection.cpp vl_objrec.cpp)
set(OBJREC_HDRS feature2d.h object_detection.h vl_objrec.h )

add_library( objrec ${OBJREC_SRCS} )
target_link_libraries( objrec keypoints ${OpenCV_LIBS} )

if(Boost_program_options_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_excutable(test_object_detection src/test_object_detection.cpp)
  target_link_libraries(test_object_detection objrec keypoints ${OpenCV_LIBS} ${Boost_LIBRARIES} )
endif()

install(TARGETS objrec
  EXPORT VisionlabToolboxTargets
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT archive
  COMPONENT dev
)

install(FILES ${OBJREC_HDRS}
  DESTINATION "${INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
  COMPONENT dev
)