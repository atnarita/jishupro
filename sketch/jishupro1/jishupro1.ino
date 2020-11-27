#define TRIG_PIN 6
#define ECHO_PIN 5
#define LIGHT_PIN_LONG 7
#define LIGHT_PIN_SHORT 8

void setup()
{
  Serial.begin(9600);
  Serial.println("start");
  pinMode(ECHO_PIN,INPUT);
  pinMode(TRIG_PIN,OUTPUT);
  pinMode(LIGHT_PIN_LONG,OUTPUT);
  pinMode(LIGHT_PIN_SHORT,OUTPUT);
}

void loop()
{
  digitalWrite(TRIG_PIN,LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN,HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN,LOW);
  int distance = pulseIn(ECHO_PIN,HIGH);
  distance=distance/57;
  if(distance<=50){
    digitalWrite(LIGHT_PIN_SHORT,HIGH);
    digitalWrite(LIGHT_PIN_LONG,LOW);
  }else if(50<=distance && distance<=55){
    digitalWrite(LIGHT_PIN_LONG,HIGH);
    digitalWrite(LIGHT_PIN_SHORT,HIGH);
  }else{
    digitalWrite(LIGHT_PIN_LONG,HIGH);
    digitalWrite(LIGHT_PIN_SHORT,LOW);
  }
  Serial.println(distance);
  delay(60);
}
