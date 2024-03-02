#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <string>

class timer
{
public:
   timer() : m_start(std::chrono::high_resolution_clock::now())
   {
   }

   ~timer()
   {
      stop();
   }

   void stop()
   {
      auto endTime = std::chrono::high_resolution_clock::now();
      auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_start).time_since_epoch().count();
      auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTime).time_since_epoch().count();

      auto duration = end - start;
      double ms = duration * 0.001;
      std::cout << duration << "us (" << ms << "ms)\n";
   }

private:
   std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};