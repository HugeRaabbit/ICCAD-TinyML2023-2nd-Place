/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"



#include <math.h>
#include "decision_tree_params.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

uint32_t getCurrentMicros(void)
{
	LL_SYSTICK_IsActiveCounterFlag();
	uint32_t m = HAL_GetTick();
	const uint32_t tms = SysTick->LOAD + 1;
	__IO uint32_t u = tms - SysTick->VAL;
	if (LL_SYSTICK_IsActiveCounterFlag())
	{
		m = HAL_GetTick();
		u = tms - SysTick->VAL;
	}
	return (m*1000+(u*1000)/tms);
}

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
float input[1][1250][1]={0};
float result[2]={0};


float std_dev2(float a[], int n) {
    if(n == 0)
        return 0.0;
    float sum = 0;
    float sq_sum = 0;
    for(int i = 0; i < n; i+=sample_size) {
       sum += a[i];
       sq_sum += a[i] * a[i];
    }
    float mean = sum *one_by_n ; 
    float variance =(sq_sum * one_by_n) - mean * mean;
    return sqrt(variance);
}


int  extract_feature_peaks(float X_data[])
{
    int len_data = 1250;
    float data_std = std_dev2(X_data, len_data) * sigma;
    int count_num  = 0;
    int last_peak = -1;
    int peaks[200]; 
    for (int i =0; i < len_data; i+=sample_size)
    {
        if( fabs(X_data[i])> data_std)
        {
            peaks[count_num] = i;
            count_num+=1;
        }

    }

    int new_peak_indices[200];

    int j = 0;
    int curr_peak_index = 0;
    last_peak = peaks[0];
    for (int t =1; t < count_num; t++)
    {
            
        int peak_indices = peaks[t]; 

        int curr_diff =   peak_indices - last_peak;
        if (curr_diff > window)
        {
            new_peak_indices[curr_peak_index] = last_peak;
            curr_peak_index++;
            last_peak = peak_indices;
        }  
        else
        {
            if (X_data[peak_indices] > X_data[last_peak])
            {
                last_peak = peak_indices;
            }
        }

    }

    if (last_peak !=-1)
    {
        new_peak_indices[curr_peak_index]=last_peak;
        curr_peak_index++;
    }

    if (curr_peak_index <3)
    {
        return 1;
    }

    int min_int = 1250; 
    int diff = 0; 
    int sum_int = 0;
    for(int i=1;i<curr_peak_index; i++)
    {
        sum_int+=new_peak_indices[i] - new_peak_indices[i-1];
        diff = new_peak_indices[i] - new_peak_indices[i-1];
        min_int = (min_int > diff) ? diff : min_int; 
    }

    float avg_int =(float)  ((float) sum_int/  (float)(curr_peak_index-1));

    if (avg_int <= decision_f1) {
        return 1; 
    }
    // else if (min_int <= decision_f1) { 
    //         return 1;
    // }
    return 0;
   
}

void aiRun(float *input, float *result)
{
    result[0] = 0.5;
    result[1]= extract_feature_peaks(input);

}





/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
  float input_tmp;
  int status=0;
  double vector[2] = {0};
  double start_time = 0;
  double end_time = 0;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_USART2_UART_Init();
  MX_TIM2_Init();
  
  /* USER CODE BEGIN 2 */
  // Model/detection method initialization function, to be implemented by teams
  
  
  
  // HugeRabbit_We don't use the Neural Networks so we don't use the part of Model_Init()
  //Model_Init();




  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

	// Send the code "begin" to tell the upper computer that it can start sending a set of test data.
    printf("begin");
    for(int16_t i=0;i<1;i++)
    {
      for(int16_t j=0;j<1250;j++)
      {
        for(int16_t k=0;k<1;k++)
        {
          // Receive test data from serial port and save to input[]
          scanf("%f",&input_tmp);
          input[i][j][k]=input_tmp;
        }

      }
    }
    HAL_Delay(1);
    //Run the model and get results
	  start_time = getCurrentMicros();

	// Model inference function, to be implemented by team
    aiRun(input,result);
	//HAL_Delay(0);
	  end_time = getCurrentMicros();
    // Send the code "ok" to tell the upper computer that inference is done and ready to send results.
    printf("ok");
    HAL_Delay(1);
    // Wait for the upper computer to send status 200, indicating that the code "ok" has been received
    scanf("%d",&status);
    if(status==200)
    {
      // Sending results to the upper computer
      if(result[0]>result[1])
        printf("0,%.6f", (double)(end_time-start_time)/(double)1000000);
      else
        printf("1,%.6f", (double)(end_time-start_time)/(double)1000000);
      HAL_Delay(30);
    }
	  
  //MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
  }
//  if(HAL_UART_Receive_IT(&huart2, (uint8_t *)&aRxBuffer,RXBUFFERSIZE)!-HAL_OK)
//  {
//	__HAL_UART_ENABLE_IT(&huart2, UART_IT_ERR);
//  }
//  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL16;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
