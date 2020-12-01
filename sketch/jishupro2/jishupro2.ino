void setup()
{
  Serial.begin(9600);
  Serial.println("start");
}

void loop()
{
  long x, y, z;
  x = analogRead(3);
  y = analogRead(4);
  z = analogRead(5);
  Serial.print(x);
  Serial.print(" ");
  Serial.print(y);
  Serial.print(" ");
  Serial.println(z);
  delay(50);
}
