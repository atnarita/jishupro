#define TRIG_PIN 5
#define ECHO_PIN 6
#define BUZZER_PIN 3

long duration, distance, x, y, z;
unsigned long previousMillius = 0;

void setup()
{
  Serial.begin(9600);
  //Serial.println("start");
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop()
{
  // 1 loop takes 100[ms]
  
  // acceleration(x, y, z)
  x = analogRead(4);
  y = analogRead(5);
  z = analogRead(6);
 
  
   // calculate distance
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  duration = pulseIn(ECHO_PIN, HIGH);
  double v = (331.5 + 0.6 * 25)/10000;
  if(duration>0) {
    distance = (duration * v)/2; // ultrasonic speed is 340m/s = 0.034cm/us
  }

  // 40  < distance        ... do nothing
  // 15 <= distance <= 40  ...intermittent sound
  //       distance < 15   ... continuing sound
  if(distance > 40) {
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  } else if (distance <= 40 && 15 <= distance) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100-100*(distance-15)/25);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100*(distance-15)/25);
  } else {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100);
  }
  Serial.println((String)x+" "+y+" "+z+" "+distance);
}
