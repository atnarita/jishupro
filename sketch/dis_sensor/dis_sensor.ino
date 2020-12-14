// HC-SR04 Ultrasonic sensor
// https://create.arduino.cc/projecthub/Isaac100/getting-started-with-the-hc-sr04-ultrasonic-sensor-036380

#define TRIG_PIN  9
#define ECHO_PIN 10
#define BUZZER_PIN 6

float duration, distance;

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {

  // calculate distance
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  duration = pulseIn(ECHO_PIN, HIGH);
  if(duration>0) {
    distance = (duration*0.0343)/2; // ultrasonic speed is 340m/s = 0.034cm/us
    Serial.print(duration);
    Serial.print(" us ");
    Serial.print(distance);
    Serial.println(" cm");
  }
  delay(200);

  if(distance > 50) {
    Serial.println("outside");
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  } else if (distance <= 50) {
    digitalWrite(BUZZER_PIN, HIGH);
    Serial.println("inside");
    delay(10);
    digitalWrite(BUZZER_PIN, LOW);
    delay(distance*10);
  }

  Serial.println("");
}
