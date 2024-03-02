#include "kNN.hpp"

/* TODO: You can implement methods, functions that support your data structures here.
 * */

//////////////////////////////////////////////////////////
// Dataset //
//////////////////////////////////////////////////////////

Dataset::Dataset() : m_nRows(0), m_nCols(0), m_nColsOfImage(0) {}

Dataset::Dataset(const Vector<Vector<int>> &data, const Vector<std::string>& header,
        int nRows, int nCols, int nColsOfImage) : m_data(data), m_header(header), m_nRows(nRows), m_nCols(nCols), m_nColsOfImage(nColsOfImage)
{
}

Dataset::Dataset(const Dataset &other)
{
   other.getShape(m_nRows, m_nCols);
   m_nColsOfImage = other.getColsOfImage();
   m_header = other.getHeader();
   m_data = other.getData();
}

Dataset &Dataset::operator=(const Dataset &other)
{
   if (this != &other)
   {
      other.getShape(m_nRows, m_nCols);
      m_nColsOfImage = other.getColsOfImage();
      m_header = other.getHeader();
      m_data = other.getData();
   }
   return *this;
}


bool Dataset::loadFromCSV(const char *fileName)
{
   if (m_nRows != 0)
   {
      m_nRows = 0;
      m_nColsOfImage = 0;
      m_nCols = 0;
      m_header.clear();
      m_data.clear();
   }

   ifstream file(fileName);
   if (!file.is_open())
      return false;

   std::string line;
   std::getline(file, line);
   std::stringstream ss(line);
   std::string token;
   while (std::getline(ss, token, ',')) 
   {
      m_header.push_back_object(token);
      m_nCols++;
   }

   std::string lastLabel = m_header.back();
   for (int i = 0; lastLabel[i] != 'x'; i++)
   {
      m_nColsOfImage = m_nColsOfImage * 10 + (lastLabel[i] - '0');
   }

   while (std::getline(file, line))
   {
      Vector<int> row;
      row.reserve(m_nCols);

      std::stringstream ss(line);
      while (std::getline(ss, token, ','))
      {
         row.push_back(std::stoi(token));
      }
      m_data.push_back_object(row);
      m_nRows++;
   }
   return true;
}

void Dataset::printHead(int nRows, int nCols) const
{
   if (nRows < 0 || nCols < 0)
      return;

   if (nRows > m_nRows)
      nRows = m_nRows;

   if (nCols > m_nCols)
      nCols = m_nCols;

   std::stringstream ss;
   ss << m_header[0];
   for (int i = 1; i < nCols; i++)
      ss << ' ' << m_header[i];

   ss << '\n';

   for (int i = 0; i < nRows; i++)
   {
      ss << m_data[i][0];
      for (int j = 1; j < nCols; j++)
         ss << ' ' << m_data[i][j];
      ss << '\n';
   }

   std::cout << ss.str();
}

void Dataset::printTail(int nRows, int nCols) const
{
   if (nRows < 0 || nCols < 0)
      return;

   if (nRows > m_nRows)
      nRows = m_nRows;

   if (nCols > m_nCols)
      nCols = m_nCols;

   std::stringstream ss;
   size_t startCol = m_nCols - nCols;
   ss << m_header[startCol];
   for (int i = startCol + 1; i < m_nCols; i++)
      ss << ' ' << m_header[i];

   ss << '\n';

   size_t startRow = m_nRows - nRows;
   for (int i = startRow; i < m_nRows; i++)
   {
      int j = startCol;
      ss << m_data[i][j];
      for (j++; j < m_nCols; j++)
         ss << ' ' << m_data[i][j];
      ss << '\n';
   }
   std::cout << ss.str();
}

void Dataset::getShape(int &nRows, int &nCols) const
{
   nRows = m_nRows;
   nCols = m_nCols;
}


void Dataset::columns() const
{
   std::cout << m_header;
}


bool Dataset::drop(int axis, int index, const std::string &columns)
{
   if (axis == 0) // row
   {
      if (index < 0 || index >= m_nRows)
         return false;
      m_data.remove(index);
      m_nRows--;
   }
   else if (axis == 1) // col
   {
      size_t colIndex = m_header.getIndex(columns);
      if (colIndex == -1)
         return false;

      for (int i = 0; i < m_nRows; i++)
      {
         m_data[i].remove(colIndex);
      }
      m_header.remove(colIndex);
      m_nCols--;
   }
   else
      return false;

   return true;
}

int dimensionCal(int start, int end, int total)
{
   if (end == -1)
      end = total - 1;
   return end - start + 1;
}

Vector<Vector<int>> Dataset::extractData(int startRow, int endRow, int startCol, int endCol)
{
   Vector<Vector<int>> data;
   data.reserve(endRow - startRow + 1);
   for (int i = startRow; i <= endRow; i++)
      data.push_back_object(m_data[i].extract(startCol, endCol));
   return data;
}

Dataset Dataset::extract(int startRow, int endRow, int startCol, int endCol)
{
   endRow = (endRow == -1) ? m_nRows - 1 : endRow;
   endCol = (endCol == -1) ? m_nCols - 1 : endCol;

   int nRows = dimensionCal(startRow, endRow, m_nRows);
   int nCols = dimensionCal(startCol, endCol, m_nCols);

   Vector<Vector<int>> data = extractData(startRow, endRow, startCol, endCol);
   Vector<std::string> header = m_header.extract(startCol, endCol);

   return Dataset(data, header, nRows, nCols, m_nColsOfImage);
}

// //////////////////////////////////////////////////////////
// // Dataset //
// //////////////////////////////////////////////////////////

// //////////////////////////////////////////////////////////
// // KNN //
// //////////////////////////////////////////////////////////

void kNN::fit(const Dataset &X_train, const Dataset &y_train)
{
   this->X_train = X_train;
   this->y_train = y_train;
}


Dataset kNN::predict(const Dataset &X_test)
{
   // Vector of distance
   Vector<Vector<int>> allDistances;
   int nRows, nCols;
   X_test.getShape(nRows, nCols);
   allDistances.reserve(nRows);

   for (const Vector<int> &testImage : X_test)
   {
      // inner Vector, v[0] are distances, v[1] are indices
      Vector<Vector<double>> distWithIndex;

      // for sorting indices

      int trainRows, trainCols;
      X_train.getShape(trainRows, trainCols);
      distWithIndex.reserve(trainRows);

      for (const Vector<int> &trainImage : X_train)
      {
         Vector<double> v;
         v.push_back(euclideanDistance(testImage, trainImage));
         v.push_back(distWithIndex.length());
         distWithIndex.push_back_object(v);
      }

      // sort
      mergeSort(distWithIndex, 0, distWithIndex.length() - 1);

      // find the most common label
      Vector<int> labelCount(10, 0);
      int maxLabel = -1;
      int maxCount = 0;
      for (int i = 0; i < k; i++)
      {
         int label = y_train[distWithIndex[i][1]][0];
         labelCount[label]++;
         if (labelCount[label] > maxCount) {
            maxCount = labelCount[label];
            maxLabel = label;
         }
      }
      Vector<int> v;
      v.push_back(maxLabel);

      allDistances.push_back_object(v);
   }

   Vector<std::string> header;
   header.push_back("");
   return Dataset(allDistances, header, nRows, 1, 0);
}

double kNN::score(const Dataset &y_test, const Dataset &y_pred)
{
   int nImages, one;
   y_test.getShape(nImages, one);
   int count = 0;

   for (int i = 0; i < nImages; i++)
   {
      if (y_test[i][0] == y_pred[i][0])
         count++;
   }

   return (double)count / nImages;
}

// //////////////////////////////////////////////////////////
// // KNN //
// //////////////////////////////////////////////////////////
void train_test_split(Dataset &X, Dataset &y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &y_train, Dataset &y_test)
{
   int nRows, nCols;
   X.getShape(nRows, nCols);

   X_train = X.extract(0, test_size * nRows, 0, -1);
   X_test = X.extract(test_size * nRows + 1, -1, 0, -1);

   y_train = y.extract(0, test_size * nRows, 0, 0);
   y_test = y.extract(test_size * nRows + 1, -1, 0, 0);
}

double euclideanDistance(const Image &a, const Image &b)
{
   double sum = 0;
   // exclude label
   for (int i = 1; i < a.length(); i++)
   {
      int diff = a[i] - b[i];
      sum += diff * diff;
   }

   double distance = sqrt(sum);
   return distance;
}


void merge(Vector<Vector<double>> &v, int start, int mid, int end)
{
   Vector<Vector<double>> left = v.extract(start, mid);
   Vector<Vector<double>> right = v.extract(mid + 1, end);

   int i = 0, j = 0, k = start;
   while (i < left.length() && j < right.length())
   {
      if (left[i][0] <= right[j][0])
         v[k++] = left[i++];
      else
         v[k++] = right[j++];
   }

   while (i < left.length())
      v[k++] = left[i++];

   while (j < right.length())
      v[k++] = right[j++];
}


void mergeSort(Vector<Vector<double>> &v, int start, int end)
{
   if (start < end)
   {
      int mid = start + (end - start) / 2;
      mergeSort(v, start, mid);
      mergeSort(v, mid + 1, end);
      merge(v, start, mid, end);
   }
}

