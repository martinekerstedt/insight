#ifndef MNIST_H
#define MNIST_H

#include <Matrix/matrix.h>
#include <Matrix/vector.h>
#include <fstream>
#include <iostream>

class MNIST
{
public:

    void print_image(Matrix image)
    {
        for (size_t i = 0; i < image.rows(); ++i) {

            for (size_t j = 0; j < image.cols(); ++j) {

                real val = image(i,j);

                if (val > 99) {
                    std::cout << val << " ";
                } else if (val > 9) {
                    std::cout << val << "  ";
                } else {
                    std::cout << val << "   ";
                }
            }

            std::cout << std::endl;
        }
    }

    Matrix read_label_file(std::string filePath)
    {
        char *mem_ptr = read_mnist_file(filePath);

        // File type and size
        unsigned int magic_number = ((unsigned int*)mem_ptr)[0];
        unsigned int number_of_items = ((unsigned int*)mem_ptr)[1];

        if (magic_number != 2049) {
            delete[] mem_ptr;
            THROW_ERROR("Invalid magic_number: " + std::to_string(magic_number));
        }


        Matrix labels_matrix(0, 10);

        for (unsigned int i = 8; i < number_of_items + 8; ++i) {

            Vector vec;
            vec.reserve(number_of_items);

            for (unsigned j = 0; j < 10; ++j) {
                if (j == (unsigned int)mem_ptr[i]) {
                    vec.pushBack(1);
                } else {
                    vec.pushBack(0);
                }
            }

            labels_matrix.addRow(vec);
        }

        delete[] mem_ptr;

        return labels_matrix;
    }

    Matrix read_image_file(std::string filePath)
    {
        char *mem_ptr = read_mnist_file(filePath);

        // File type and size
        unsigned int magic_number = ((unsigned int*)mem_ptr)[0];
        unsigned int number_of_images = ((unsigned int*)mem_ptr)[1];
        unsigned int number_of_rows = ((unsigned int*)mem_ptr)[2];
        unsigned int number_of_cols = ((unsigned int*)mem_ptr)[3];

        if (magic_number != 2051) {
            delete[] mem_ptr;
            THROW_ERROR("Invalid magic_number: " + std::to_string(magic_number));
        }


        Matrix images_mat(0, number_of_rows*number_of_cols);

        for (unsigned int i = 0; i < number_of_images; ++i) {

            Vector image;
            image.reserve(number_of_rows*number_of_cols);

            for (unsigned int j = 0; j < number_of_rows; ++j) {
                for (unsigned int k = 0; k < number_of_cols; ++k) {

                    unsigned int mem_index = (i*number_of_rows*number_of_cols) + (j*number_of_cols) + k + 16;
                    image.pushBack((unsigned char)mem_ptr[mem_index]);
                }

            }

            images_mat.addRow(image);
        }

        delete[] mem_ptr;

        return images_mat;
    }

private:
    void swapByteOrder(unsigned int& ui)
    {
        ui = (ui >> 24) |
             ((ui << 8) & 0x00FF0000) |
             ((ui >> 8) & 0x0000FF00) |
             (ui << 24);
    }

    char* read_mnist_file(std::string filePath)
    {
        std::ifstream file;
        file.open(filePath, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            THROW_ERROR("Failed to open file");
        }

        std::streampos size;
        size = file.tellg();

        char* mem_ptr;
        mem_ptr = new char[size];

        file.seekg(0, std::ios::beg);
        file.read(mem_ptr, size);
        file.close();

        // Check endianess
        short int number = 0x1;
        char* numPtr = (char*)&number;

        if (numPtr[0] == 1) {

            // Is little endian
            swapByteOrder(((unsigned int*)mem_ptr)[0]);
            swapByteOrder(((unsigned int*)mem_ptr)[1]);

            // Is a image file
            if (((unsigned int *)mem_ptr)[0] == 2051) {

                swapByteOrder(((unsigned int*)mem_ptr)[2]);
                swapByteOrder(((unsigned int*)mem_ptr)[3]);
            }
        }

        return mem_ptr;
    }

};

#endif // MNIST_H
