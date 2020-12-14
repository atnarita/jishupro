void setup()
{
  Serial.begin(9600);
  //Serial.println("start");
}

void loop()
{
  long x, y, z;
  x = analogRead(4);
  y = analogRead(5);
  z = analogRead(6);
  Serial.println((String)x+" "+y+" "+z);
  delay(100);
}
