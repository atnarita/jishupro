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
  Serial.print("X:");
  Serial.print(x);
  Serial.print("\tY:");
  Serial.print(y);
  Serial.print("\tZ:");
  Serial.println(z);
  delay(50);
}
