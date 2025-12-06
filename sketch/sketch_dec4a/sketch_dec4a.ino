#include <Servo.h>

Servo servo_x;
Servo servo_y;

// 舵机角度范围限制 (根据你的机械结构实际情况调整)
// X轴 (水平): 通常 0-180 都可以，如果有线缆限制可以改为 20-160
const int SERVO_X_MIN = 0;   
const int SERVO_X_MAX = 180; 

// Y轴 (垂直): 垂直方向最容易撞到底座，建议限制在 45-135 之间
const int SERVO_Y_MIN = 45;  
const int SERVO_Y_MAX = 135; 

int face_center_x = -1;
int face_center_y = -1;

const int frame_center_x = 320; // 假设摄像头分辨率为640 * 480
const int frame_center_y = 240;

// 水平舵机PID参数
float Kp_x = 0.8, Ki_x = 0.0, Kd_x = 0.1;
// 垂直舵机PID参数
float Kp_y = 0.6, Ki_y = 0.0, Kd_y = 0.08;

float error_x =0, last_error_x =0, integral_x = 0;
float error_y =0, last_error_y =0, integral_y =0;

float servo_x_angle = 85.0;
float servo_y_angle = 90.0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial1.begin(115200); 
  Serial1.setTimeout(50);

  servo_x.attach(13);
  servo_y.attach(12);
  servo_x.write(servo_x_angle);
  servo_y.write(servo_y_angle);
}

void loop() {
  // put your main code here, to run repeatedly:
  receiveFacePosition();
  if (face_center_x != -1 && face_center_y != -1) {
    pidServoControl();

    face_center_x = -1;
    face_center_y = -1;
  }

  delay(20);
}



void receiveFacePosition() {
  if (Serial1.available() > 0) {
    String data = Serial1.readStringUntil('\n');
    data.trim();
    Serial.print(data);
    if (data.indexOf("RESET") != -1) {
      servo_x_angle = 85.0;
      servo_y_angle = 90.0;
      servo_x.write((int)servo_x_angle);
      servo_y.write((int)servo_y_angle);

      integral_x = 0; 
      last_error_x = 0;
      integral_y = 0; 
      last_error_y = 0;

      face_center_x = -1;
      face_center_y = -1;

      return; 
    }

    if (data.length() > 0 && data.indexOf(',') != -1) {
      
      int firstCommaIndex = data.indexOf(',');
      int secondCommaIndex = data.indexOf(',', firstCommaIndex + 1);

      if (firstCommaIndex != -1 && secondCommaIndex != -1) {
        String xStr = data.substring(firstCommaIndex + 1, secondCommaIndex);
        String yStr = data.substring(secondCommaIndex + 1);


        face_center_x = xStr.toInt();
        face_center_y = yStr.toInt();

      }
    }
  }
}


/**
 * 二自由度PID控制舵机
 */
void pidServoControl() {
  error_x = face_center_x - frame_center_x;
  integral_x += error_x;
  float derivative_x = error_x - last_error_x;
  float output_x = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative_x;

  if (abs(error_x) > 10) {
    servo_x_angle += output_x / 50.0;
    servo_x_angle = constrain(servo_x_angle, SERVO_X_MIN, SERVO_X_MAX);
    servo_x.write((int)servo_x_angle);
  }
  last_error_x = error_x;

  error_y = face_center_y - frame_center_y;
  integral_y += error_y;
  float derivative_y = error_y - last_error_y;
  float output_y = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative_y;


  if (abs(error_y) > 10) {
    servo_y_angle -= output_y / 50.0;
    servo_y_angle = constrain(servo_y_angle, SERVO_Y_MIN, SERVO_Y_MAX);
    servo_y.write((int)servo_y_angle);
  }
  last_error_y = error_y;
}