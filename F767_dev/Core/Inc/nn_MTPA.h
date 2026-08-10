/*
 * nn.h
 * Author: Jaehoon Shim
 */

#ifndef NN_H
#define NN_H

#include "main.h"
#define HIDDEN_NUM 20
#define SYS_CLOCK_HZ          (96.0f * 1.0e6f)
#define TICK_TO_NS            (1.0e9f / SYS_CLOCK_HZ)
#define TICK_TO_US            (1.0e6f / SYS_CLOCK_HZ)


#define SYS_CLOCK_HZ    (96.0f * 1.0e6f)
#define TICK_TO_US      (1.0e6f / SYS_CLOCK_HZ)

#define CALC_PERIOD(x)                                              \
    do                                                              \
    {                                                               \
        (x).period_us =                                             \
            (float)(DWT->CYCCNT - (x).tickStart) * TICK_TO_US;      \
    } while (0)

#define TICK_START(x)                                               \
    do                                                              \
    {                                                               \
        (x).tickStart = DWT->CYCCNT;                                \
    } while (0)

#define START_TICK_MANAGER(x)                                       \
    do                                                              \
    {                                                               \
        CALC_PERIOD(x);                                             \
        TICK_START(x);                                              \
        (x).cnt++;                                                  \
    } while (0)

#define TICK_STOP_CHECK(x)                                          \
    do                                                              \
    {                                                               \
        (x).elapsedTime_us =                                        \
            (float)(DWT->CYCCNT - (x).tickStart) * TICK_TO_US;      \
    } while (0)

#define TICK_MAX_CHECK(x)                                           \
    do                                                              \
    {                                                               \
        if ((x).elapsedTime_us > (x).maxElapsedTime_us)             \
        {                                                           \
            (x).maxElapsedTime_us = (x).elapsedTime_us;             \
        }                                                           \
    } while (0)

#define PERIOD_MAX_CHECK(x)                                         \
    do                                                              \
    {                                                               \
        if ((x).period_us > (x).maxPeriod_us)                        \
        {                                                           \
            (x).maxPeriod_us = (x).period_us;                        \
        }                                                           \
    } while (0)

#define END_TICK_MANAGER(x)                                         \
    do                                                              \
    {                                                               \
        TICK_STOP_CHECK(x);                                         \
        TICK_MAX_CHECK(x);                                          \
        PERIOD_MAX_CHECK(x);                                        \
    } while (0)


typedef struct
{
    uint32_t cnt;
    uint32_t tickStart;

    __IO float elapsedTime_us;
    __IO float maxElapsedTime_us;
    __IO float period_us;
    __IO float maxPeriod_us;

} TimeManager;

extern TimeManager ControlTime;
extern TimeManager NN_Time1;
extern TimeManager NN_Time2;
extern TimeManager NN_Time3;
extern TimeManager NN_Time4;

extern volatile float Id_MTPA;
extern volatile float Iq_MTPA;
extern volatile float hidden_output[HIDDEN_NUM];
extern volatile float nn_input[1];

/* Te 입력으로 Id, Iq를 계산합니다. */
void NN_Run(volatile float *x);

#endif /* NN_H */


