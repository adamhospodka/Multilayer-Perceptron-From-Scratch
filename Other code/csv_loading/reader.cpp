#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>

#define print(x)           \
    {                      \
        cout << x << endl; \
    };
#define type(x)                                \
    {                                          \
        std::cout << typeid(x).name() << '\n'; \
    }
#define array_size(x) (sizeof(x) / sizeof(*x))

using namespace std;

class fileHandler
{
public:
    int labels_count;  // = 10;
    int img_width;     // = 28;
    int img_height;    //; = 28;
    int img_dimesions; // = img_width * img_height;

    fileHandler()
    {
        labels_count = 10;
        img_width = 28;
        img_height = 28;
        img_dimesions = img_width * img_height;
    }

    string status()
    {
        return string("status - alive!\n");
    }

    std::vector<vector<int> > load_labels(string file, int examples)
    {

        ifstream soubor(file, std::ifstream::in);

        std::string content;
        // Seek to 0 characters from the beginning of the file
        soubor.seekg(0, std::ios::end);
        // String content -> make as big as "get position of pointer in stream"
        content.resize(soubor.tellg());
        // Offset from the beginning of the stream’s buffer)
        soubor.seekg(0, std::ios::beg);
        // Read from beginning to the end
        soubor.read(&content[0], content.size());
        // Close stream
        soubor.close();

        //cout << content;
        //type(content);


        int counter = 0;

        vector<vector<int> > collection;
        vector<int> featureBuffer;

        int ahoj = 0;

        size_t b = 0;
        while (true)
        {
            int x = std::stoi(content, &b, 10);
            //print(x);

            // Counter control ----------------------
            featureBuffer.push_back(x);
            counter++;

            if (counter == 10) // features columns
            {
                print("--");
                collection.push_back(featureBuffer);
                counter = 0;
                featureBuffer.clear();
            }

            
            // File conrol ---------------------------
            if (b >= content.size())
            {
                break;
            }
            
            content = content.substr(b + 1);



        }

        print(collection[3][0]);
        print(collection[3][1]);
        print(collection[3][2]);
        print(collection[3][3]);
        print(collection[3][4]);
        print(collection[3][5]);
        print(collection[3][6]);
        print(collection[3][7]);
        print(collection[3][8]);
        print(collection[3][9]);

        return collection;
    };
};

int main()
{
    cout << "\n\nRunning...";
    fileHandler csvReader;
    cout << csvReader.status();
    vector<vector<int> > data = csvReader.load_labels("file.csv", 2);
 
    print(data[0][1])
}
