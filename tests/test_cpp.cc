#include <iostream>
#include <vector>

using namespace std;

void print_vec(vector<int*> &vec) {
   for (auto &itr: vec)
      if (itr == NULL)
         cout << itr << endl;
}

int main() {
   vector<int*> kps;
   kps = vector<int*>(10, NULL);
   print_vec(kps);
}
 