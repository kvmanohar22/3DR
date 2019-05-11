//
// Created by kv on 10/5/19.
//

#include <iostream>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/StdVector>

using namespace std;
using namespace Eigen;

int main() {

    Vector3d vec;
    vec << 2.0, 3.0, 1.0;
    cout << vec.transpose() << endl;

    // Access the first two elements
    cout << vec.head<2>() << endl;

    Vector3d vnor = vec.normalized();
    cout << "Normalized: " << vnor.transpose() << endl;

    cout << vnor.head<3>().transpose() / vnor[2] << endl;
}
