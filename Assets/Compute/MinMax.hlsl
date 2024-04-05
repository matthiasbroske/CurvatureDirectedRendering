#ifndef MIN_MAX_INCLUDED
#define MIN_MAX_INCLUDE

struct MinMax
{
    float min;
    float max;
};

struct QuantizedMinMax
{
    int min;
    int max;
};

int Quantize(float x)
{
    return (int) (x * 1024.);
}

float Unquantize(int x)
{
    return x / 1024.;
}

#endif