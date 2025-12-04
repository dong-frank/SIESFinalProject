#include <Servo.h>

Servo servo_x;
Servo servo_y;

int face_center_x = -1;
int face_center_y = -1;

//TODO
const int frame_center_x = 320; // 假设摄像头分辨率为640 * 480
const int frame_center_y = 240;

// 水平舵机PID参数
float Kp_x = 0.5, Ki_x = 0.0, Kd_x = 0.1;
// 垂直舵机PID参数
float Kp_y = 0.4, Ki_y = 0.0, Kd_y = 0.08;

float error_x =0, last_error_x =0, integral_x = 0;
float error_y =0, last_error_y =0, integral_y =0;

int servo_x_angle = 90;
int servo_y_angle = 90;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  //TODO
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
  }
}


void receiveFacePosition() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.println("收到数据: " + data);

    int cStart = data.indexOf("center=(");
    int cEnd = data.indexOf(")", cStart);
    if (cStart != -1 && cEnd != -1) {
      String centerStr = data.substring(cStart + 8, cEnd);
      int commaIdx = centerStr.indexOf(",");
      if (commaIdx != -1) {
        face_center_x = centerStr.substring(0, commaIdx).toInt();
        face_center_y = centerStr.substring(commaIdx + 1).toInt();
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
    servo_x_angle -= output_x / 50;
    servo_x_angle = constrain(servo_x_angle, 0, 180);
    servo_x.write(servo_x_angle);
  }
  last_error_x = error_x;

  error_y = face_center_y - frame_center_y;
  integral_y += error_y;
  float derivative_y = error_y - last_error_y;
  float output_y = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative_y;

  if (abs(error_y) > 10) {
    servo_y_angle -= output_y / 50;
    servo_y_angle = constrain(servo_y_angle, 0, 180);
    servo_y.write(servo_y_angle);
  }
  last_error_y = error_y;
}