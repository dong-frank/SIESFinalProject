
int face_center_x = -1;
int face_center_y = -1;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  receiveFacePosition();
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
        Serial.print("中心点X: "); Serial.println(face_center_x);
        Serial.print("中心点Y: "); Serial.println(face_center_y);
      }
    }
  }
}
