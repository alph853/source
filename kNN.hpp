#include "main.hpp"

/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */


template <typename T>
class Vector;
using Image = Vector<int>;

template <typename T>
class SLL;


template <typename T>
std::ostream &operator<<(std::ostream &os, const Vector<T> &vector);

template <typename T>
std::ostream &operator<<(std::ostream &os, const SLL<T> &List);

void mergeSort(Vector<Vector<double>> &v, int start, int end);
void merge(Vector<Vector<double>> &v, int start, int mid, int end);

template <typename T>
class List
{
public:
    virtual ~List() = default;
    virtual void push_back(T value) = 0;
    virtual void push_front(T value) = 0;
    virtual void insert(int index, T value) = 0;
    virtual void remove(int index) = 0;
    virtual T &get(int index) const = 0;
    virtual int length() const = 0;
    virtual void clear() = 0;
    virtual void print() const = 0;
    virtual void reverse() = 0;
};

template <typename T>
class Vector : public List<T>
{
public:
    Vector();
    Vector(size_t size, const T &value);
    ~Vector();
    Vector(const Vector &other);
    void push_back(T value) override;
    void push_front(T value) override;
    void insert(int index, T value) override;
    void remove(int index) override;
    T &get(int index) const override;
    int length() const override;
    void clear() override;
    void print() const override;
    void reverse() override;

    void push_back_object(const T &value);
    void push_front_object(const T &value);
    void insert_object(int index, const T &value);

    void reserve(size_t amount = 0);
    T back() const { return m_array[m_size - 1]; }
    bool isEmpty() const { return m_size == 0; }
    void ensureCapacity();
    size_t capacity() const { return m_capacity; }
    size_t getIndex(const T &obj) const;


    T &operator[](int index);
    const T &operator[](int index) const;

    Vector<T> &operator=(const Vector<T> &other);
    friend std::ostream &operator<< <>(std::ostream &os, const Vector<T> &vector);
    Vector<T> extract(int start, int end) const;

    T *begin() { return m_array; }
    T *end() { return m_array + m_size; }
    const T *begin() const { return m_array; }
    const T *end() const { return m_array + m_size; }

private:
    size_t m_size;
    size_t m_capacity;
    T *m_array;
};

template <typename T>
class SLL : public List<T>
{
private:
    struct Node
    {
        Node() : next(nullptr){};
        Node(T data, Node *next = nullptr) : m_data(data), next(next) {}

        T m_data;
        Node *next;
    };

    Node *head;
    Node *tail;
    size_t m_size;

public:
    SLL(Node *head = nullptr, Node *tail = nullptr, size_t size = 0) : head(head), tail(tail), m_size(size)
    {
    }

    virtual void push_back(T value) override;
    virtual void push_front(T value) override;
    virtual void insert(int index, T value) override;
    virtual void remove(int index) override;
    T &get(int index) const override;
    int length() const override;
    void clear() override;
    void print() const override;
    void reverse() override;

    friend std::ostream &operator<< <>(std::ostream &os, const SLL<T> &List);
};


class Dataset
{
public:
    Dataset();

    Dataset(const Vector<Vector<int>> &data, const Vector<std::string>& header,
            int nRows, int nCols, int nColsOfImage);

    ~Dataset() {};
    Dataset(const Dataset &other);
    Dataset &operator=(const Dataset &other);
    bool loadFromCSV(const char *fileName);
    void printHead(int nRows = 5, int nCols = 5) const;
    void printTail(int nRows = 5, int nCols = 5) const;
    void getShape(int &nRows, int &nCols) const;
    void columns() const;
    bool drop(int axis = 0, int index = 0, const std::string &columns = "");

    Dataset extract(int startRow = 0, int endRow = -1, int startCol = 0, int endCol = -1);
    Vector<Vector<int>> extractData(int startRow, int endRow, int startCol, int endCol);

    const Vector<Vector<int>>& getData() const { return m_data; }
    const Vector<std::string>& getHeader() const { return m_header; }

    Vector<int> &operator[](int index) { return m_data[index]; }
    const Vector<int> &operator[](int index) const { return m_data[index]; }

    Vector<int> *begin() { return m_data.begin(); }
    Vector<int> *end() { return m_data.end(); }
    const Vector<int> *begin() const { return m_data.begin(); }
    const Vector<int> *end() const { return m_data.end(); }
    int getColsOfImage() const { return m_nColsOfImage; }

private:
    int m_nRows;
    int m_nCols;
    int m_nColsOfImage;
    Vector<Vector<int>> m_data;
    Vector<std::string> m_header;
};

class kNN
{
private:
    int k;
    Dataset X_train;
    Dataset y_train;
    // You may need to define more
public:
    kNN(int k = 5) : k(k) {}
    void fit(const Dataset &X_train, const Dataset &y_train);
    Dataset predict(const Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred);
};

void train_test_split(Dataset &X, Dataset &y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &y_train, Dataset &y_test);

double euclideanDistance(const Vector<int> &a, const Vector<int> &b);

//////////////////////////////////////////////////////////
// VECTOR //
//////////////////////////////////////////////////////////

template <typename T>
Vector<T>::Vector() : m_size(0), m_capacity(10), m_array(new T[m_capacity])
{
}

template <typename T>
Vector<T>::Vector(size_t size, const T &value) : m_size(size), m_capacity(size), m_array(new T[size])
{
    for (int i = 0; i < size; i++)
    {
        m_array[i] = value;
    }
}

template <typename T>
Vector<T>::Vector(const Vector &other) : m_size(other.length()), m_capacity(other.capacity()), m_array(new T[m_capacity])
{
    for(int i = 0; i < m_size; i++)
        m_array[i] = other[i];
}

template <typename T>
Vector<T> &Vector<T>::operator=(const Vector<T> &other) {
    if (this != &other)
    {
        delete[] m_array;
        m_size = other.length();
        m_capacity = other.capacity();
        m_array = new T[m_capacity];
        for(int i = 0; i < m_size; i++) 
            m_array[i] = other[i];
    }
    return *this;
}


template <typename T>
void Vector<T>::push_back(T value)
{
    this->ensureCapacity();
    m_array[m_size++] = value;
}


template <typename T>
void Vector<T>::push_back_object(const T& value)
{
    this->ensureCapacity();
    m_array[m_size++] = value;
}

template <typename T>
void Vector<T>::push_front(T value)
{
    this->insert(0, value);
}

template <typename T>
void Vector<T>::push_front_object(const T& value)
{
    this->insert(0, value);
}

template <typename T>
void Vector<T>::insert(int index, T value)
{
    this->ensureCapacity();

    if (index < 0 || index > m_size)
        return;

    for(int i = m_size; i > index; i--)
        m_array[i] = m_array[i - 1];
    m_array[index] = value;
    m_size++;
}

template <typename T>
void Vector<T>::insert_object(int index, const T& value)
{
    this->ensureCapacity();

    if (index < 0 || index > m_size)
        return;

    for (int i = m_size; i > index; i--)
        m_array[i] = m_array[i - 1];
    m_array[index] = value;
    m_size++;
}

template <typename T>
void Vector<T>::ensureCapacity()
{
    if (m_size == m_capacity)
    {
        m_capacity *= 1.5;
        T *newArray = new T[m_capacity];
        for (int i = 0; i < m_size; i++) {
            newArray[i] = m_array[i];
        }
        delete[] m_array;
        m_array = newArray;
    }
}

template <typename T>
size_t Vector<T>::getIndex(const T &obj) const
{
    for (size_t i = 0; i < m_size; i++)
    {
        if (m_array[i] == obj)
            return i;
    }
    return -1;
}

template <typename T>
void Vector<T>::reserve(size_t amount)
{
    if (amount > m_capacity)
    {
        m_capacity = amount;
        T *newArray = new T[m_capacity];
        for (int i = 0; i < m_size; i++)
            newArray[i] = m_array[i];
        delete[] m_array;
        m_array = newArray;
    }
}

template <typename T>
int Vector<T>::length() const
{
    return m_size;
}

template <typename T>
void Vector<T>::reverse()
{
    for (size_t i = 0; i < m_size / 2; i++)
    {
        std::swap(m_array[i], m_array[m_size - i - 1]);
    }
}

template <typename T>
void Vector<T>::remove(int index)
{
    if (this->isEmpty() || index < 0 || index >= m_size)
        return;

    for (size_t i = index; i < m_size - 1; i++)
        m_array[i] = m_array[i + 1];
    m_size--;
}

template <typename T>
T &Vector<T>::get(int index) const
{
    if (index < 0 || index >= m_size)
        throw std::out_of_range("Out of range");
    return m_array[index];
}

template <typename T>
T &Vector<T>::operator[](int index)
{
    if (index < 0 || index >= m_size)
        throw std::out_of_range("Out of range");
    return m_array[index];
}

template <typename T>
const T &Vector<T>::operator[](int index) const
{
    if (index < 0 || index >= m_size)
        throw std::out_of_range("Out of range");
    return m_array[index];
}


template <typename T>
std::ostream &operator<<(ostream &os, const Vector<T> &vector)
{
    if (vector.m_size != 0)
    {
        os << vector.m_array[0];
        for (size_t i = 1; i < vector.m_size; i++)
            os << ' ' << vector.m_array[i];
        os << '\n';
    }
    return os;
}

template <typename T>
void Vector<T>::clear()
{
    m_size = 0;
    delete[] m_array;
    m_capacity = 10;
    m_array = new T[m_capacity];
}

template <typename T>
Vector<T>::~Vector()
{
    delete[] m_array;
}

template <typename T>
void Vector<T>::print() const
{
    std::cout << *this;
}

template <typename T>
Vector<T> Vector<T>::extract(int start, int end) const
{
    if (start == 0 && end == m_size - 1)
        return *this;

    Vector<T> res;
    res.reserve(end - start + 1);
    for (size_t i = start; i <= end; i++)
        res.push_back(m_array[i]);
    return res;
}
//////////////////////////////////////////////////////////
// VECTOR //
//////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
// LIST //
//////////////////////////////////////////////////////////

template <typename T>
void SLL<T>::push_back(T value)
{
    Node *newNode = new Node(value);
    if (head == nullptr)
    {
        head = newNode;
        tail = newNode;
    }
    else
    {
        tail->next = newNode;
        tail = newNode;
    }
    m_size++;
}

template <typename T>
void SLL<T>::push_front(T value)
{
    Node *newNode = new Node(value);
    if (head == nullptr)
    {
        head = newNode;
        tail = newNode;
    }
    else
    {
        newNode->next = head;
        head = newNode;
    }
    m_size++;
}

template <typename T>
void SLL<T>::insert(int index, T value)
{
    if (index < 0 || index > m_size)
    {
        return;
    }
    if (index == 0)
    {
        push_front(value);
    }
    else if (index == m_size)
    {
        push_back(value);
    }
    else
    {
        Node *newNode = new Node(value);
        Node *current = head;
        for (int i = 0; i < index - 1; i++)
        {
            current = current->next;
        }
        newNode->next = current->next;
        current->next = newNode;
        m_size++;
    }
}

template <typename T>
void SLL<T>::remove(int index)
{
    if (index < 0 || index >= m_size)
    {
        return;
    }
    if (index == 0)
    {
        Node *temp = head;
        head = head->next;
        delete temp;
    }
    else
    {
        Node *current = head;
        for (int i = 0; i < index - 1; i++)
        {
            current = current->next;
        }
        Node *temp = current->next;
        current->next = temp->next;
        delete temp;
    }
    m_size--;
}

template <typename T>
T &SLL<T>::get(int index) const
{
    if (index < 0 || index >= m_size)
    {
        throw std::out_of_range("Out of range");
    }
    Node *current = head;
    for (int i = 0; i < index; i++)
    {
        current = current->next;
    }
    return current->m_data;
}

template <typename T>
int SLL<T>::length() const
{
    return m_size;
}

template <typename T>
void SLL<T>::clear()
{
    Node *current = head;
    while (current != nullptr)
    {
        Node *temp = current;
        current = current->next;
        delete temp;
    }
    head = nullptr;
    tail = nullptr;
    m_size = 0;
}

template <typename T>
void SLL<T>::print() const
{
    if (m_size == 0)
    {
        std::cout << '\n';
        return;
    }

    std::stringstream ss;
    Node *current = head;
    ss << current->m_data;

    current = current->next;

    while (current != nullptr)
    {
        ss << ' ' << current->m_data;
        current = current->next;
    }
    std::cout << ss.str() << '\n';
}

template <typename T>
void SLL<T>::reverse()
{
    Node *prev = nullptr;
    Node *current = head;
    Node *nextNode = nullptr;

    while (current)
    {
        nextNode = current->next;
        current->next = prev;

        prev = current;
        current = nextNode;
    }
    head = prev;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const SLL<T> &List)
{
    List.print();
    return os;
}

//////////////////////////////////////////////////////////
// LIST //
//////////////////////////////////////////////////////////
