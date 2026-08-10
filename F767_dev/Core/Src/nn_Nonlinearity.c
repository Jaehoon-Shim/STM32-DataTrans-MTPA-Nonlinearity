/*
 * nn.c
 * Author: Jaehoon Shim
 */
#include "nn_Nonlinearity.h"
/* =========================================================
 * Live Expression variables
 * ========================================================= */

/* Check these outputs in Live Expressions. */
volatile float Vcomp_NN = 0.0f;

/* Optional variables for debugging. */
volatile float hidden1_output[HIDDEN1_NUM];
volatile float hidden2_output[HIDDEN2_NUM];

/* =========================================================
 * normalization parameters
 * ========================================================= */
/* Input Te MINMAX */
const float I_BASE = 4.6f;



/* =========================================================
 * NN parameters
 * ========================================================= */
/* Auto-generated PyTorch ANN parameters */
/* Weight layout: [input][output] */

/* Auto-generated PyTorch ANN parameters */
/* Weight layout: [input][output] */

const float nn_theta1[1][12] = {
    {0.7183036f, 1.3193697f, -1.0744791f, 1.055324f, -0.3938918f, 0.4317314f, 0.f, 0.712129f, 1.4065877f, -1.7979144f, 1.3546176f, -0.1024862f}
};

const float nn_bias1[12] = {0.734301f, 0.0519307f, 0.2766932f, -0.3196781f, 0.4269858f, 0.1604881f, 0.f, 0.317105f, 0.0866989f, 0.2272554f, 0.0985211f, 0.2712339f};

const float nn_theta2[12][12] = {
    {0.1477975f, 0.1855692f, 0.f, -0.0072949f, 0.4019221f, 0.1722683f, 0.f, -0.1462048f, -0.0000022f, 0.f, -0.0636006f, 0.0315558f},
    {0.0598291f, 0.3530784f, 0.f, -0.0012292f, -0.027214f, 0.0651762f, 0.f, -1.28522f, 0.f, 0.f, -0.006106f, 0.0010507f},
    {-0.4734235f, -0.1031818f, 0.f, 0.0006937f, -0.3505533f, -0.1187122f, 0.f, 0.8456008f, 0.f, 0.f, 0.0036679f, -0.0017563f},
    {-0.8622135f, -0.2865152f, 0.f, 0.f, -0.3842627f, -0.2558546f, 0.f, 0.f, 0.f, 0.f, 0.0027925f, -0.0269294f},
    {0.1157401f, 0.1554362f, 0.f, 0.003021f, -0.0776927f, -0.0247685f, 0.f, 0.3930905f, 0.f, 0.f, 0.0382918f, 0.0703988f},
    {0.0428378f, 0.0465788f, 0.f, -0.0000204f, 0.196565f, 0.0521935f, 0.f, -0.3706683f, 0.f, 0.f, -0.0063019f, -0.0039967f},
    {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
    {0.1024655f, 0.3656116f, 0.f, -0.003091f, 0.3522403f, 0.0594302f, 0.f, -0.3656521f, 0.f, 0.f, -0.0359286f, -0.1214461f},
    {-0.0681279f, 0.1899878f, 0.f, 0.f, -0.0270037f, 0.0529347f, 0.f, -1.4210997f, 0.f, 0.f, 0.f, 0.0000016f},
    {-0.402633f, -0.4859458f, 0.f, 0.f, -0.5294028f, -0.1165279f, 0.f, 1.4313434f, 0.f, 0.f, 0.f, 0.f},
    {0.0746888f, 0.3049528f, 0.f, 0.f, 0.2971744f, 0.0148703f, 0.f, -1.351524f, 0.f, 0.f, 0.0000011f, -0.0001861f},
    {0.0866057f, 0.1531868f, 0.f, 0.003534f, 0.007923f, -0.0234083f, 0.f, 0.1428768f, 0.f, 0.f, -0.0853409f, 0.0692386f}
};

const float nn_bias2[12] = {0.1468827f, -0.030269f, 0.f, -0.0152019f, 0.2666706f, 0.0542264f, 0.f, 0.2507847f, -0.0000027f, 0.f, 0.0459532f, -0.2152336f};

const float nn_theta3[12][1] = {
    {0.5166886f},
    {0.3264767f},
    {0.f},
    {-0.0003615f},
    {0.3667952f},
    {0.1902101f},
    {0.f},
    {-1.6849558f},
    {0.0075717f},
    {0.f},
    {0.0351619f},
    {0.0016895f}
};

const float nn_bias3[1] = {0.3278485f};




/* =========================================================
 * NN calculation
 * ========================================================= */
void NN_Run(volatile float *x)
{
    int i;
    int j;
    float y_output;
    float x_pu;


    START_TICK_MANAGER(NN_Time1);
    x_pu = x[0] / I_BASE;
    for (i = 0; i < HIDDEN1_NUM; i++)
    {
        hidden1_output[i] = x_pu * nn_theta1[0][i] + nn_bias1[i];
        if (hidden1_output[i] < 0.0f)
        {
            hidden1_output[i] = 0.0f;
        }
    }
    END_TICK_MANAGER(NN_Time1);


    START_TICK_MANAGER(NN_Time2);
    for (i = 0; i < HIDDEN2_NUM; i++)
    {
        hidden2_output[i] = nn_bias2[i];
        for (j = 0; j < HIDDEN1_NUM; j++)
        {
            hidden2_output[i] += hidden1_output[j] * nn_theta2[j][i];
        }
        if (hidden2_output[i] < 0.0f)
        {
            hidden2_output[i] = 0.0f;
        }
    }
    END_TICK_MANAGER(NN_Time2);


    START_TICK_MANAGER(NN_Time3);
    y_output = nn_bias3[0];
    for (i = 0; i < HIDDEN2_NUM; i++)
    {
        y_output += hidden2_output[i] * nn_theta3[i][0];
    }
    Vcomp_NN = y_output;
    END_TICK_MANAGER(NN_Time3);
}




