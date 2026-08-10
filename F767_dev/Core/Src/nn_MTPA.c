/*
 * nn.c
 * Author: Jaehoon Shim
 */
#include "nn_MTPA.h"
/* =========================================================
 * Live Expression variables
 * ========================================================= */

/* Check these outputs in Live Expressions. */
volatile float Id_MTPA = 0.0f;
volatile float Iq_MTPA = 0.0f;

/* Optional variables for debugging. */
volatile float hidden_output[HIDDEN_NUM];

/* =========================================================
 * MINMAX parameters
 * ========================================================= */
/* Input Te MINMAX */
const float x0_min = 0.87078f;
const float x0_max = 261.94f;

/* Output Id, Iq MINMAX */
const float y0_min = -93.615f;
const float y0_max = -0.0086464f;

const float y1_min = 1.0f;
const float y1_max = 285.0f;


/* =========================================================
 * NN parameters
 * ========================================================= */
/* Auto-generated PyTorch ANN parameters */
/* Weight layout: [input][output] */

const float nn_theta1[1][20] = {
    {0.f, 0.4526574f, 0.3816075f, 0.f, 0.9350858f, 0.3594839f, -0.4519671f, 0.4000468f, -0.4457596f, 0.f, 0.5679298f, 0.7424262f, -0.5791892f, -0.2239582f, 0.742748f, -0.6074338f, 0.f, -0.000012f, -0.4734888f, 0.f}
};

const float nn_bias1[20] = {0.f, -0.2660788f, 0.5187801f, 0.f, -0.4274435f, 0.0571478f, 0.3177628f, 0.0341797f, 0.44761f, -0.0000003f, 0.2859695f, 0.8810407f, 0.5806002f, 0.2284946f, -0.064853f, 0.1848244f, 0.f, -0.000186f, 0.6113898f, 0.f};

const float nn_theta2[20][2] = {
    {0.f, 0.f},
    {-0.2442885f, 0.000464f},
    {0.0320996f, -0.0035018f},
    {0.f, 0.f},
    {-0.3987152f, 0.1057947f},
    {0.023861f, 0.3138882f},
    {-0.4580906f, -0.0427677f},
    {-0.1626543f, 0.0423065f},
    {0.5892984f, -0.1752026f},
    {0.f, 0.f},
    {-0.0301919f, -0.035544f},
    {0.2827172f, 0.258498f},
    {0.4519554f, -0.3237044f},
    {0.3506425f, -0.1174118f},
    {-0.4661306f, 0.3556979f},
    {-0.4695565f, -0.2788404f},
    {0.f, 0.f},
    {0.0000332f, -0.000051f},
    {0.4232236f, -0.0615567f},
    {0.f, 0.f}
};

const float nn_bias2[2] = {0.1187976f, 0.1721375f};




/* =========================================================
 * NN calculation
 * ========================================================= */
void NN_Run(volatile float *x)
{
    int i;
    float x0_normalized;
    float y0, y1, y0_normalized, y1_normalized;

    /* 1. Input normalization */
    START_TICK_MANAGER(NN_Time1);
    x0_normalized = (x[0] - x0_min) / (x0_max - x0_min);
    END_TICK_MANAGER(NN_Time1);


    /* 2. Hidden layer + ReLU */
    START_TICK_MANAGER(NN_Time2);
    for (i = 0; i < HIDDEN_NUM; i++)
    {
        hidden_output[i] = x0_normalized * nn_theta1[0][i] + nn_bias1[i];

        if (hidden_output[i] < 0.0f)
        {
            hidden_output[i] = 0.0f;
        }
    }
    END_TICK_MANAGER(NN_Time2);

    /* 3. Output layer */
    START_TICK_MANAGER(NN_Time3);
    y0_normalized = nn_bias2[0];
    y1_normalized = nn_bias2[1];
    for (i = 0; i < HIDDEN_NUM; i++)
    {
    	y0_normalized += hidden_output[i] * nn_theta2[i][0];
    	y1_normalized += hidden_output[i] * nn_theta2[i][1];
    }
    END_TICK_MANAGER(NN_Time3);

    /* 4. Output denormalization */
    START_TICK_MANAGER(NN_Time4);
    y0 = y0_normalized * (y0_max - y0_min) + y0_min;
    y1 = y1_normalized * (y1_max - y1_min) + y1_min;

    Id_MTPA = y0;
    Iq_MTPA = y1;
    END_TICK_MANAGER(NN_Time4);
}



//void NN_Run(volatile float *x)
//{
//    int i;
//    float x0_normalized;
//    float y0, y1, y0_normalized, y1_normalized;
//
//    /* 1. Input normalization */
//    x0_normalized = (x[0] - x0_min) / (x0_max - x0_min);
//
//    /* 2. Hidden layer + ReLU */
//    for (i = 0; i < HIDDEN_NUM; i++)
//    {
//        hidden_output[i] = x0_normalized * nn_theta1[0][i] + nn_bias1[i];
//
//        if (hidden_output[i] < 0.0f)
//        {
//            hidden_output[i] = 0.0f;
//        }
//    }
//
//    /* 3. Output layer */
//    y0_normalized = nn_bias2[0];
//    y1_normalized = nn_bias2[1];
//    for (i = 0; i < HIDDEN_NUM; i++)
//    {
//    	y0_normalized += hidden_output[i] * nn_theta2[i][0];
//    	y1_normalized += hidden_output[i] * nn_theta2[i][1];
//    }
//
//    /* 4. Output denormalization */
//    y0 = y0_normalized * (y0_max - y0_min) + y0_min;
//    y1 = y1_normalized * (y1_max - y1_min) + y1_min;
//
//    Id_MTPA = y0;
//    Iq_MTPA = y1;
//}




