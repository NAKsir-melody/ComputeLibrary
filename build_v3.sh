#aarch64-linux-android-clang++ examples/graph_inception_v3.cpp utils/Utils.cpp utils/GraphUtils.cpp -I. -Iinclude -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -Lbuild -L. -o graph_inception_v3_aarch64 -static-libstdc++ -pie -DARM_COMPUTE_CL
#aarch64-linux-android-clang++ examples/graph_inception_v3.cpp utils/Utils.cpp utils/GraphUtils.cpp -I. -Iinclude -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -Lbuild -L. -o graph_inception_v3_aarch64 -static-libstdc++ -pie -DARM_COMPUTE_CL
#aarch64-linux-android-clang++ examples/graph_alexnet.cpp utils/Utils.cpp utils/GraphUtils.cpp -I. -Iinclude -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -Lbuild -L. -o graph_alexnet_aarch64 -static-libstdc++ -pie -DARM_COMPUTE_CL
aarch64-linux-android-clang++ examples/graph_vgg19.cpp utils/Utils.cpp utils/GraphUtils.cpp -I. -Iinclude -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -Lbuild -L. -o graph_vgg19_aarch64 -static-libstdc++ -pie -DARM_COMPUTE_CL



