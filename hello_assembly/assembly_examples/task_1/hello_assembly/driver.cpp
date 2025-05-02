#include <iostream>

extern "C" void hello_assembly();

using namespace std;

int main() {
    hello_assembly();
    return 0;
}
