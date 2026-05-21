import { NextResponse } from 'next/server';
import { BACKEND_URL, USE_LIVE_BACKEND } from '@/lib/backendConfig';

export async function POST(request: Request) {
  const payload = await request.json();

  if (!USE_LIVE_BACKEND) {
    return NextResponse.json({ status: 'logged_mock', id: `MOCK_${Date.now()}` });
  }

  try {
    const res = await fetch(`${BACKEND_URL}/outcome`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    const data = text ? JSON.parse(text) : {};

    if (!res.ok) {
      return NextResponse.json(
        { status: 'error', detail: data.detail ?? `Backend returned ${res.status}` },
        { status: res.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.warn('Outcome backend unavailable; returning mock logged response.', error);
    return NextResponse.json({ status: 'logged_mock', id: `MOCK_${Date.now()}` });
  }
}
