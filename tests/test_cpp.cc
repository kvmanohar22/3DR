#include <iostream>
#include <vector>

#include <sophus/se3.h>

using namespace std;
using namespace Sophus;

void print_vec(vector<int*> &vec) {
   for (auto &itr: vec)
      if (itr == NULL)
         cout << itr << endl;
}

int main() {
   vector<int*> kps;
   kps = vector<int*>(10, NULL);
   print_vec(kps);

    // Sophus
    SE3 Tfw = SE3();
    cout << Tfw.rotation_matrix() << endl;
    cout << Tfw.translation() << endl;
}
 