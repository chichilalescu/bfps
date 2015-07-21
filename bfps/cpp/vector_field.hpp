#include "field_descriptor.hpp"

template <class rnumber>
class vector_field
{
    private:
        field_descriptor<rnumber> *descriptor;
        rnumber *rdata;
        rnumber (*cdata)[2];
        bool is_real;
    public:
        vector_field(field_descriptor<rnumber> *d, rnumber *data);
        vector_field(field_descriptor<rnumber> *d, rnumber (*data)[2]);
        ~vector_field();

        /* various operators */
        vector_field &operator*(rnumber factor);
};

